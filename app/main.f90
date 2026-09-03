
program main
    use, intrinsic :: iso_fortran_env, only : int64
    use shuffle_mod, only : run_shuffle_ensemble
    use precision_mod
    use parameter_mod
    use random_mod
    use network_mod
    use langevin_mod
    use statistics_mod
    use output_mod
    use theory_mod
    use energetics_mod
    implicit none

    integer :: i
    integer :: relax_percent, sample_percent
    integer, parameter :: max_wall_steps = 16
    integer :: n_wall_steps
    logical :: q_is_upper, q_is_lower
    real(dp) :: wall_program_time
    real(dp) :: max_real_part, max_lyapunov_residual
    real(dp) :: triangular_tol
    integer(int64) :: wall_program_start, wall_program_end, wall_clock_rate
    integer(int64) :: wall_step_start
    character(len=48) :: wall_step_labels(max_wall_steps)
    real(dp) :: wall_step_times(max_wall_steps)

    integer, allocatable :: layer_sizes(:)
    integer :: nstep, n_relax, n_hidden
    real(dp) :: max_fixedpoint_residual, max_force_at_fixedpoint
    logical, allocatable :: adj_matrix(:,:)
    real(dp), allocatable :: r(:), W(:, :), noise(:, :)
    real(dp), allocatable :: x(:), y(:), force(:), Q(:, :), fixpoint(:)
    
    real(dp), allocatable :: force_at_fixedpoint(:)
    real(dp), allocatable :: mean_x(:), mean_force(:), K0(:, :), Ktau(:, :)
    real(dp), allocatable :: K0_theory(:, :)
    real(dp), allocatable :: alpha(:, :), alpha_sim(:, :)
    
    integer :: n_bias
    character(len=16) :: resolved_bias_mode
    integer, allocatable :: bias_layer(:)
    real(dp), allocatable :: bias(:)

    real(dp), allocatable :: y_next(:)
    real(dp), allocatable :: heat_rate(:), work_rate(:), entropy_rate(:), internal_rate(:)
    real(dp), allocatable :: heat_rate_theory(:), work_rate_theory(:)
    real(dp), allocatable :: internal_rate_theory(:), entropy_rate_theory(:)

    type(SimulationParameters) :: param
    type(StatisticsState) :: stat
    type(EnergeticsState) :: energy

    call system_clock(wall_program_start, wall_clock_rate)
    if (wall_clock_rate <= 0_int64) error stop "system_clock rate is invalid"
    wall_step_start = wall_program_start
    n_wall_steps = 0
    wall_step_labels = ""
    wall_step_times = 0.0_dp

    call read_parameters("input/parameters.nml", param)
    call record_wall_step("Read parameters", wall_step_start, wall_clock_rate)

    print *, "-------------------------------"
    print *, "Dynamics parameters"
    print *, "-------------------------------"

    print *, "r mean      = ", param%r_mean
    print *, "r std       = ", param%r_std
    print *, "weight mean = ", param%weight_mean
    print *, "weight std  = ", param%weight_std
    print *, "coupling type = ", trim(param%coupling_type)

    print *, "-------------------------------"
    print *, "Noise parameters"
    print *, "-------------------------------"

    print *, "sigma mean = ", param%sigma_mean

    print *, "-------------------------------"
    print *, "Simulation parameters"
    print *, "-------------------------------"

    print *, "run_simulation = ", param%run_simulation
    if (param%run_simulation) then
    print *, "dt       = ", param%dt
    print *, "t relax  = ", param%t_relax
    print *, "t sample = ", param%t_sample
    print *, "lag steps = ", param%lag_steps
    print *, "seed     = ", param%seed
    endif
    ! Initialize the random number generator with the specified seed
    call initialize_seed(param%seed)

    select case (trim(adjustl(param%graph_type)))

    case default
    error stop "Unsupported graph type: "//trim(param%graph_type)

    case("ER")
      call generate_er(param, adj_matrix, W)

    case("FCNN")      
      n_hidden = 2
      allocate(layer_sizes(n_hidden+2))      
      layer_sizes = 100; layer_sizes(1) = 800; layer_sizes(n_hidden+2) = 10
      param%N = sum(layer_sizes)
      param%directed = .true.
      call generate_FCNN(param, n_hidden, layer_sizes, adj_matrix, W)

    case ("EXTERNAL")

      call read_weighted_edge_list(trim(param%network_file), &
        adj_matrix, W, param%N)

      param%directed = .true.

    end select
    call record_wall_step("Build/read network", wall_step_start, wall_clock_rate)

    print *, "-------------------------------"
    print *, "Network parameters"
    print *, "-------------------------------"
    print *, "N = ", param%N
    print*, "Network density =", real(count(adj_matrix), dp) / real(param%N * (param%N - 1), dp)
    print *, "Graph type = ", trim(param%graph_type)
    print *, "Directed   = ", param%directed
    print *, "-------------------------------"

    call set_parameters(param, r, noise)
    
    call initialize_bias( &
      param, bias, bias_layer, n_bias, resolved_bias_mode)

    print *, "-------------------------------"
    print *, "Bias parameters"
    print *, "-------------------------------"
    print *, "Bias mode =", trim(resolved_bias_mode)
    print *, "Number of biases =", n_bias

    select case (trim(resolved_bias_mode))
    case ("RANDOM")
      print *, "Bias mean =", param%bias_mean
      print *, "Bias std  =", param%bias_std
    case ("FILE")
      print *, "Bias file =", trim(param%bias_file)
      print *, "No-bias nodes =", count(bias_layer == 0)
      print *, "Layer 1 count =", count(bias_layer == 1)
      print *, "Layer 2 count =", count(bias_layer == 2)
      print *, "Layer 3 count =", count(bias_layer == 3)
    end select
    print *, "-------------------------------"

    call record_wall_step("Set dynamics, noise, and bias", wall_step_start, wall_clock_rate)

    ! solve analytic covariance K0, alpha
    allocate(K0_theory(param%N, param%N))

    allocate(fixpoint(param%N))

    call construct_Q(r, W, Q)    

    call record_wall_step("Allocate theory arrays and construct Q", &
                          wall_step_start, wall_clock_rate)

    triangular_tol = 100.0_dp * epsilon(1.0_dp) * &
                     max(1.0_dp, maxval(abs(Q)))
    q_is_upper = is_upper_triangular(Q, triangular_tol)
    q_is_lower = is_lower_triangular(Q, triangular_tol)

    print *, "Q is upper triangular =", q_is_upper
    print *, "Q is lower triangular =", q_is_lower

    if (trim(adjustl(param%coupling_type)) == "DIFFUSIVE") then

    call solve_fixed_point_linear( &
        Q, bias, &
        q_is_upper, q_is_lower, &
        fixpoint)

    else

      error stop "TANH fixed-point solver is not implemented yet"

    end if

    max_fixedpoint_residual = &
    maxval(abs(matmul(Q, fixpoint) + bias))

    allocate(force_at_fixedpoint(param%N))

    call compute_force( &
    fixpoint, r, W, bias, &
    param%coupling_type, &
    force_at_fixedpoint)

    max_force_at_fixedpoint = &
    maxval(abs(force_at_fixedpoint))   

    print *, "-------------------------------"
    print *, "Fixed-point verification"
    print *, "-------------------------------"
    print *, "max |Q*x* + bias| =", &
    max_fixedpoint_residual    
     print *, "max |F(x*)| =", max_force_at_fixedpoint

    call write_node_results( &
      'output/node.csv', r, noise, fixpoint, bias)

    if (trim(adjustl(param%graph_type)) /= "EXTERNAL") then
      call write_edge_results('output/edge.csv', W, adj_matrix)
    else
      print *, "Skip edge.csv: network was read from an external file."
    end if

    call record_wall_step("Write network output", wall_step_start, wall_clock_rate)

    ! Theory
    if (q_is_upper .or. q_is_lower) then

      if (q_is_upper) then
        print *, "Lyapunov solver: upper-triangular DTRSYL3 solver"
      else
        print *, "Lyapunov solver: lower-triangular DTRSYL3 solver"
      end if

      call solve_lyapunov_triangular_blocked( &
        Q, -noise, K0_theory, max_real_part, &
        q_is_upper, q_is_lower)
      call record_wall_step("Solve Lyapunov equation (DTRSYL3)", &
                            wall_step_start, wall_clock_rate)

    else

      print *, "Lyapunov solver: general Schur + DTRSYL3 solver"
      call solve_lyapunov_blocked( &
        Q, -noise, K0_theory, max_real_part)
      call record_wall_step("Solve Lyapunov equation (Schur + DTRSYL3)", &
                            wall_step_start, wall_clock_rate)

    end if

    call analytic_result(Q, noise, K0_theory, alpha, &
                         q_is_upper, q_is_lower)
    call record_wall_step("Compute theory alpha", &
                          wall_step_start, wall_clock_rate)

    call compute_energetics_theory(Q, noise, alpha, &
    heat_rate_theory, work_rate_theory, &
    internal_rate_theory, entropy_rate_theory)
    call record_wall_step("Compute theory energetics", &
                          wall_step_start, wall_clock_rate)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (param%run_simulation) then

    nstep = int(param%t_sample / param%dt) ! no. of time steps
    n_relax = int(param%t_relax / param%dt) ! no. of burn-in steps

    call system_clock(wall_step_start)
    call initialize_state(param%N, x, y, force)

    ! burn-in period to reach steady state
    relax_percent = -1
    do i = 1, n_relax
        call compute_force(x, r, W, bias, param%coupling_type, force) ! x(t), f(t)
        call langevin_step(x, force, noise, param%dt) ! x(t+dt)
        call show_progress("Burn-in", i, n_relax, relax_percent)
    end do
    call record_wall_step("Burn-in simulation", wall_step_start, wall_clock_rate)

    call system_clock(wall_step_start)
    allocate(y_next(param%N))

    call initialize_statistics(stat, param%lag_steps, param%N)
    call initialize_energetics(energy, Q, noise)
    call record_wall_step("Initialize sampling statistics", &
                          wall_step_start, wall_clock_rate)

    call system_clock(wall_step_start)
    sample_percent = -1
    ! Langevin simulation
    do i = 1, nstep
        call compute_force(x, r, W, bias, param%coupling_type, force)
        y = x - fixpoint ! y = x - 1, y(t)
        ! statistics 使用 t 時刻的 y 與 nonlinear force
        call update_statistics(stat, y, force) ! x(t), f(t)
        ! x(t) -> x(t+dt)
        call langevin_step(x, force, noise, param%dt)
        ! t+dt 時刻
        y_next = x - fixpoint ! y(t+dt)
        ! Stratonovich midpoint energetics
        call update_energetics_linear(energy, y, y_next, param%dt) ! y(t+dt) - y(t)
        call show_progress("Sampling", i, nstep, sample_percent)
    enddo

    call record_wall_step("Sampling simulation", wall_step_start, wall_clock_rate)

    call system_clock(wall_step_start)
    call finalize_statistics(stat, fixpoint, mean_x, mean_force, K0, Ktau) ! calculate covariance, mean_x, mean_f
    call finalize_energetics(energy, heat_rate, work_rate, & ! calculate average energetic rates
                         internal_rate, entropy_rate)

    ! compute alpha with Ktau
    call simulate_alpha(Ktau, param%lag_steps*param%dt, alpha_sim) ! alpha = (K_tau.T - K_tau) / tau

    call write_mean_results('output/mean.csv', mean_x, mean_force)   
    call write_correlation_results('output/correlation.csv', K0, Ktau)    
    call write_energetics_results('output/energetics.csv', heat_rate, work_rate, &
                         internal_rate, entropy_rate)
    call record_wall_step("Finalize simulation and write output", &
                          wall_step_start, wall_clock_rate)

    print *, "-------------------------------"
    print *, "Theory verification"
    print *, "-------------------------------"
    print *, "max |K0 - K0_theory| =", maxval(abs(K0 - K0_theory))
    print *, "max |<x> - x*| =", maxval(abs(mean_x - fixpoint))
    print *, "max |alpha - alpha_sim| =", maxval(abs(alpha - alpha_sim))
    print *, "max |<F>| =", maxval(abs(mean_force))
    print *, "-------------------------------"
    print *, "Energetics (Simulation and Theory)"
    print *, "-------------------------------"
    print *, "Total heat rate =", &
    sum(heat_rate), sum(heat_rate_theory)  
    print *, "Total work rate =", &
    sum(work_rate), sum(work_rate_theory)   
    print *, "Total energy rate =", &
    sum(internal_rate), sum(internal_rate_theory)   
    print *, "Total entropy rate =", &
    sum(entropy_rate), sum(entropy_rate_theory) 

    else

    print *, "Langevin simulation skipped."    
    print *, "-------------------------------"
    print *, "Energetics (Simulation and Theory)"
    print *, "-------------------------------"
    print *, "Total heat rate =", &
    sum(heat_rate_theory)  
    print *, "Total entropy production rate =", &
    sum(entropy_rate_theory)  
    print *, "Total work rate =", &
    sum(work_rate_theory)  

    end if

    call write_energetics_results( &
    'output/energetics_theory.csv', &
    heat_rate_theory, work_rate_theory, &
    internal_rate_theory, entropy_rate_theory)

    ! call write_alpha('output/alpha.csv', alpha, alpha_sim)

    if (param%verify_lyapunov) then
      call compute_lyapunov_residual(Q, K0_theory, noise, &
                                     q_is_upper, q_is_lower, &
                                     max_lyapunov_residual)

      print *, "-------------------------------"
      print *, "max |residual of Lyapunov| =", max_lyapunov_residual
    end if
    call record_wall_step("Post-process and write theory output", &
                          wall_step_start, wall_clock_rate)

    if (param%n_weight_shuffles > 0) then

      call run_shuffle_ensemble( &
      adj_matrix, W, r, noise, &
      q_is_upper, q_is_lower, &
      param%n_weight_shuffles, &
      param%shuffle_seed, &
      sum(entropy_rate_theory), &
      "output/shuffle_stability.csv", &
      "output/shuffle_energetics.csv", &
      "output/shuffle_summary.csv")

      call record_wall_step( &
        "Weight-shuffle ensemble", &
        wall_step_start, wall_clock_rate)

    end if

    call system_clock(wall_program_end)
    wall_program_time = real(wall_program_end - wall_program_start, dp) / &
                        real(wall_clock_rate, dp)

    print *, "-------------------------------"
    print *, "Wall-clock timing"
    print *, "-------------------------------"
    do i = 1, n_wall_steps
      write(*, '(2X,A,T48,F10.3," s")') &
        trim(wall_step_labels(i)), wall_step_times(i)
    end do
    print *, "-------------------------------"
    write(*, '(2X,A,T48,F10.3," s")') &
      "Total wall-clock time", wall_program_time

contains

  subroutine record_wall_step(label, count_start, count_rate)
    character(len=*), intent(in) :: label
    integer(int64), intent(inout) :: count_start
    integer(int64), intent(in) :: count_rate

    integer(int64) :: count_end

    call system_clock(count_end)

    if (n_wall_steps >= max_wall_steps) then
      error stop "Too many wall-clock timing records"
    end if

    n_wall_steps = n_wall_steps + 1
    wall_step_labels(n_wall_steps) = label
    wall_step_times(n_wall_steps) = &
      real(count_end - count_start, dp) / real(count_rate, dp)

    count_start = count_end
  end subroutine record_wall_step

  subroutine show_progress(label, current_step, total_step, last_percent)
    use, intrinsic :: iso_fortran_env, only : output_unit
    implicit none
    character(len=*), intent(in) :: label
    integer, intent(in) :: current_step, total_step
    integer, intent(inout) :: last_percent

    integer, parameter :: bar_width = 40
    integer :: percent, filled
    character(len=bar_width) :: bar

    if (total_step <= 0) return

    percent = int(100.0_dp * real(current_step, dp) &
              / real(total_step, dp))

    percent = min(100, max(0, percent))

    ! 百分比沒有改變時不輸出
    if (percent <= last_percent .and. current_step < total_step) return

    filled = percent * bar_width / 100

    bar = repeat("#", filled) // repeat("-", bar_width - filled)

    write(output_unit, '(a,a," [",a,"] ",i3,"%")', advance="no") &
            achar(13), trim(label), bar, percent

    flush(output_unit)

    last_percent = percent

    ! 完成後換行，避免後面的輸出接在進度條後面
    if (current_step == total_step) then
      write(output_unit, *)
    end if

  end subroutine show_progress

end program main
