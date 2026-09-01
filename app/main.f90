
program main
    use, intrinsic :: iso_fortran_env, only : int64
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
    real(dp) :: cpu_program_start, cpu_program_end
    real(dp) :: cpu_relax_start, cpu_relax_end
    real(dp) :: cpu_sample_start, cpu_sample_end
    real(dp) :: wall_program_time
    real(dp) :: max_real_part
    integer(int64) :: wall_program_start, wall_program_end, wall_clock_rate
    integer(int64) :: wall_step_start

    integer, allocatable :: layer_sizes(:)
    integer :: nstep, n_relax, n_hidden
    logical, allocatable :: adj_matrix(:,:)
    real(dp), allocatable :: r(:), W(:, :), noise(:, :)
    real(dp), allocatable :: x(:), y(:), force(:), Q(:, :), fixpoint(:)
    real(dp), allocatable :: mean_x(:), mean_force(:), K0(:, :), Ktau(:, :)
    real(dp), allocatable :: K0_theory(:, :), Ktau_theory(:, :)
    real(dp), allocatable :: alpha(:, :), alpha_sim(:, :), delta(:, :)

    real(dp), allocatable :: y_next(:), residual(:, :)
    real(dp), allocatable :: heat_rate(:), work_rate(:), entropy_rate(:), internal_rate(:)
    real(dp), allocatable :: heat_rate_theory(:), work_rate_theory(:)
    real(dp), allocatable :: internal_rate_theory(:), entropy_rate_theory(:)

    type(SimulationParameters) :: param
    type(StatisticsState) :: stat
    type(EnergeticsState) :: energy

    call system_clock(wall_program_start, wall_clock_rate)
    if (wall_clock_rate <= 0_int64) error stop "system_clock rate is invalid"
    wall_step_start = wall_program_start
    call cpu_time(cpu_program_start)

    call read_parameters("input/parameters.nml", param)
    call report_wall_step("Read parameters", wall_step_start, wall_clock_rate)

    print *, "-------------------------------"
    print *, "Dynamics parameters"
    print *, "-------------------------------"

    print *, "r mean      = ", param%r_mean
    print *, "r std       = ", param%r_std
    print *, "weight mean = ", param%weight_mean
    print *, "weight std  = ", param%weight_std

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
    call report_wall_step("Build/read network", wall_step_start, wall_clock_rate)
    
    print *, "-------------------------------"
    print *, "Network parameters"
    print *, "-------------------------------"
    print *, "N = ", param%N
    print*, "Network density =", real(count(adj_matrix), dp) / real(param%N * (param%N - 1), dp)
    print *, "Graph type = ", trim(param%graph_type)
    print *, "Directed   = ", param%directed
    print *, "-------------------------------"

    call set_parameters(param, r, noise)
    call report_wall_step("Set dynamics and noise", wall_step_start, wall_clock_rate)

    call write_node_results('output/node.csv', r, noise) 
    call write_edge_results('output/edge.csv', W, adj_matrix)
    call report_wall_step("Write network output", wall_step_start, wall_clock_rate)

    ! solve analytic covariance K0, alpha
    allocate(K0_theory(param%N, param%N), Ktau_theory(param%N, param%N), delta(param%N, param%N))
    allocate(alpha(param%N, param%N), alpha_sim(param%N, param%N))

    allocate(fixpoint(param%N))
    fixpoint = 1.0_dp

    call construct_Q(r, W, Q)    
    call report_wall_step("Allocate theory arrays and construct Q", &
                          wall_step_start, wall_clock_rate)

    print *, "Q is upper triangular =", &
    is_upper_triangular(Q, 1.0e-12_dp)

    print *, "Q is lower triangular =", &
    is_lower_triangular(Q, 1.0e-12_dp)
    
    ! call write_jacobian_results('output/Q.csv', Q)

    ! Theory
    if (is_upper_triangular(Q, 1.0e-12_dp)) then

      print *, "Lyapunov solver: upper-triangular direct solver"
      call solve_lyapunov_triangular( &
        Q, -noise, K0_theory, max_real_part)

    else if (is_lower_triangular(Q, 1.0e-12_dp)) then

      print *, "Lyapunov solver: lower-triangular direct solver"
      call solve_lyapunov_triangular( &
        Q, -noise, K0_theory, max_real_part)

    else

      print *, "Lyapunov solver: general Schur solver"
      call solve_lyapunov( &
        Q, -noise, K0_theory, max_real_part)

    end if
    call report_wall_step("Solve Lyapunov equation", &
                          wall_step_start, wall_clock_rate)

    call analytic_result(Q, noise, K0_theory, param%lag_steps*param%dt, & ! Solve delta, alpha, Ktau
        delta, alpha, Ktau_theory)
    call report_wall_step("Compute theory correlations", &
                          wall_step_start, wall_clock_rate)

    call compute_energetics_theory(Q, noise, alpha, &
    heat_rate_theory, work_rate_theory, &
    internal_rate_theory, entropy_rate_theory)
    call report_wall_step("Compute theory energetics", &
                          wall_step_start, wall_clock_rate)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if (param%run_simulation) then

    nstep = int(param%t_sample / param%dt) ! no. of time steps
    n_relax = int(param%t_relax / param%dt) ! no. of burn-in steps
  
    call system_clock(wall_step_start)
    call initialize_state(param%N, x, y, force)

    call cpu_time(cpu_relax_start)
    ! burn-in period to reach steady state
    relax_percent = -1
    do i = 1, n_relax
        call compute_force(x, r, W, force) ! x(t), f(t)
        call langevin_step(x, force, noise, param%dt) ! x(t+dt)
        call show_progress("Burn-in", i, n_relax, relax_percent)
    end do
    call cpu_time(cpu_relax_end)
    call report_wall_step("Burn-in simulation", wall_step_start, wall_clock_rate)

    call system_clock(wall_step_start)
    allocate(y_next(param%N))

    call initialize_statistics(stat, param%lag_steps, param%N)
    call initialize_energetics(energy, Q, noise)
    call report_wall_step("Initialize sampling statistics", &
                          wall_step_start, wall_clock_rate)

    call system_clock(wall_step_start)
    call cpu_time(cpu_sample_start)
    sample_percent = -1
    ! Langevin simulation
    do i = 1, nstep
        call compute_force(x, r, W, force)
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

    call cpu_time(cpu_sample_end)
    call report_wall_step("Sampling simulation", wall_step_start, wall_clock_rate)

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
    call report_wall_step("Finalize simulation and write output", &
                          wall_step_start, wall_clock_rate)

    print *, "-------------------------------"
    print *, "Theory verification"
    print *, "-------------------------------"
    print *, "max |K0 - K0_theory| =", maxval(abs(K0 - K0_theory))
    print *, "max |Ktau - Ktau_theory| =", maxval(abs(Ktau - Ktau_theory))
    print *, "max |x - x*| after relaxation =", maxval(abs(x - fixpoint))
    print *, "max |alpha - alpha_sim| =", maxval(abs(alpha - alpha_sim))

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

    ! call write_correlation_results( &
    ! 'output/correlation_theory.csv', &
    ! K0_theory, Ktau_theory)

    call write_energetics_results( &
    'output/energetics_theory.csv', &
    heat_rate_theory, work_rate_theory, &
    internal_rate_theory, entropy_rate_theory)
    
    ! call write_alpha('output/alpha.csv', alpha, alpha_sim)

    print *, "-------------------------------"
    residual = matmul(Q, K0_theory) + matmul(K0_theory,transpose(Q)) + noise
    print *, "max |residue of Lyapunov| =", maxval(abs(residual))   
    call report_wall_step("Post-process and write theory output", &
                          wall_step_start, wall_clock_rate)

    call cpu_time(cpu_program_end)
    call system_clock(wall_program_end)
    wall_program_time = real(wall_program_end - wall_program_start, dp) / &
                        real(wall_clock_rate, dp)

    print *, "-------------------------------"
    print *, "CPU timing"
    print *, "-------------------------------"
    if (param%run_simulation) then
        write(*, '("Burn-in CPU time : ",f12.3," s")') &
        cpu_relax_end - cpu_relax_start
        write(*, '("Sampling CPU time: ",f12.3," s")') &
        cpu_sample_end - cpu_sample_start
    end if
    write(*, '("Total CPU time   : ",f12.3," s")') &
    cpu_program_end - cpu_program_start
    write(*, '("Wall-clock time  : ",f12.3," s")') wall_program_time

contains

  subroutine report_wall_step(label, count_start, count_rate)
    character(len=*), intent(in) :: label
    integer(int64), intent(inout) :: count_start
    integer(int64), intent(in) :: count_rate

    integer(int64) :: count_end
    real(dp) :: elapsed

    call system_clock(count_end)
    elapsed = real(count_end - count_start, dp) / real(count_rate, dp)

    write(*, '("Wall-clock | ",A,": ",F10.3," s")') &
      trim(label), elapsed

    count_start = count_end
  end subroutine report_wall_step

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
