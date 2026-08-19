
program main
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

    logical, allocatable :: adj_matrix(:,:)
    real(dp), allocatable :: r(:), W(:, :), noise(:, :)
    real(dp), allocatable :: x(:), y(:), force(:), Q(:, :)
    real(dp), allocatable :: mean_x(:), mean_force(:), K0(:, :), Ktau(:, :)
    real(dp), allocatable :: K0_theory(:, :), Ktau_theory(:, :)
    real(dp), allocatable :: alpha(:, :), alpha_sim(:, :), delta(:, :)
    integer :: nstep, n_relax

    real(dp), allocatable :: y_next(:)
    real(dp), allocatable :: heat_rate(:), work_rate(:), entropy_rate(:), internal_rate(:)
    real(dp), allocatable :: heat_rate_theory(:), work_rate_theory(:)
    real(dp), allocatable :: internal_rate_theory(:), entropy_rate_theory(:)

    type(SimulationParameters) :: param
    type(StatisticsState) :: stat
    type(EnergeticsState) :: energy

    call cpu_time(cpu_program_start)

    call read_parameters("input/parameters.nml", param)

    print *, "-------------------------------"
    print *, "Network parameters"
    print *, "-------------------------------"

    print *, "Graph type = ", trim(param%graph_type)
    print *, "Directed   = ", param%directed
    print *, "N          = ", param%N
    print *, "p          = ", param%p

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

    print *, "dt       = ", param%dt
    print *, "t relax  = ", param%t_relax
    print *, "t sample = ", param%t_sample
    print *, "lag steps = ", param%lag_steps
    print *, "seed     = ", param%seed

    ! Initialize the random number generator with the specified seed
    call initialize_seed(param%seed)

    call set_parameters(param, r, noise)

    print*, "-------------------------------"
    print*, "Mean r     =", sum(r) / real(param%N, dp)
    print*, "Mean noise =", sum(noise) / real(param%N, dp)
    print*, "-------------------------------"   

    call generate_er(param, adj_matrix, W)
    print*, "Network density =", real(count(adj_matrix), dp) / real(param%N * (param%N - 1), dp)

    ! solve analytic covariance K0, alpha
    allocate(K0_theory(param%N, param%N), Ktau_theory(param%N, param%N), delta(param%N, param%N))
    allocate(alpha(param%N, param%N), alpha_sim(param%N, param%N))

    call construct_Q(r, W, Q)    
    
    call solve_lyapunov(Q, -noise, K0_theory) ! Solve K0, alpha 

    call analytic_result(Q, noise, K0_theory, param%lag_steps*param%dt, &
        delta, alpha, Ktau_theory)

    call compute_energetics_theory(Q, noise, alpha, &
    heat_rate_theory, work_rate_theory, &
    internal_rate_theory, entropy_rate_theory)

    call write_edge_results('output/edge.csv', W, adj_matrix)
    call write_jacobian_results('output/Q.csv', Q)

    nstep = int(param%t_sample / param%dt) ! no. of time steps
    n_relax = int(param%t_relax / param%dt) ! no. of burn-in steps

    call initialize_state(param%N, x, y, force)

    call cpu_time(cpu_relax_start)
    ! burn-in period to reach steady state
    relax_percent = -1
    do i = 1, n_relax
        call compute_force(x, r, W, force)
        call langevin_step(x, force, noise, param%dt)
        call show_progress("Burn-in", i, n_relax, relax_percent)
    end do
    call cpu_time(cpu_relax_end)

    allocate(y_next(param%N))

    call initialize_statistics(stat, param%lag_steps, param%N)
    call initialize_energetics(energy, Q, noise)

    call cpu_time(cpu_sample_start)
    sample_percent = -1
    ! Langevin simulation
    do i = 1, nstep
        call compute_force(x, r, W, force)
        y = x - 1.0_dp ! y = x - 1
        ! statistics 使用 t 時刻的 y 與 nonlinear force
        call update_statistics(stat, y, force)
        ! x(t) -> x(t+dt)
        call langevin_step(x, force, noise, param%dt)
        ! t+dt 時刻
        y_next = x - 1.0_dp
        ! Stratonovich midpoint energetics
        call update_energetics_linear(energy, y, y_next, param%dt)
        call show_progress("Sampling", i, nstep, sample_percent)
    enddo

    call cpu_time(cpu_sample_end)

    ! allocate(heat_rate(param%N), work_rate(param%N))
    ! allocate(internal_rate(param%N), entropy_rate(param%N))

    call finalize_statistics(stat, mean_x, mean_force, K0, Ktau) ! calculate covariance, mean_x, mean_f
    call finalize_energetics(energy, heat_rate, work_rate, &
                         internal_rate, entropy_rate)

    ! compute alpha with Ktau
    call simulate_alpha(Ktau, param%lag_steps*param%dt, alpha_sim)

    call write_node_results('output/node.csv', r, noise, mean_x, mean_force)    
    call write_correlation_results('output/correlation.csv', K0, Ktau)    
    call write_energetics_results('output/energetics.csv', heat_rate, work_rate, &
                         internal_rate, entropy_rate)
    ! call write_alpha('output/alpha.csv', alpha, alpha_sim)

    print *, "max |K0 - K0_theory| =", maxval(abs(K0 - K0_theory))
    print *, "max |Ktau - Ktau_theory| =", maxval(abs(Ktau - Ktau_theory))
    print *, "max |x - 1| after relaxation =", maxval(abs(x - 1.0_dp))
    print *, "max |alpha - alpha_sim| =", maxval(abs(alpha - alpha_sim))

    print *, "-------------------------------"
    print *, "Energetics"
    print *, "-------------------------------"
    print *, "max heat error =", &
    maxval(abs(heat_rate - heat_rate_theory))
    print *, "max work error =", &
    maxval(abs(work_rate - work_rate_theory))
    print *, "max internal error =", &
    maxval(abs(internal_rate - internal_rate_theory))
    print *, "max entropy error =", &
    maxval(abs(entropy_rate - entropy_rate_theory))
    print *, "Theory total entropy rate =", &
    sum(entropy_rate_theory)

    call cpu_time(cpu_program_end)

    print *, "-------------------------------"
    print *, "CPU timing"
    print *, "-------------------------------"
    write(*, '("Burn-in CPU time : ",f12.3," s")') &
    cpu_relax_end - cpu_relax_start
    write(*, '("Sampling CPU time: ",f12.3," s")') &
    cpu_sample_end - cpu_sample_start
    write(*, '("Total CPU time   : ",f12.3," s")') &
    cpu_program_end - cpu_program_start

contains

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