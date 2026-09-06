program test_linearized_energetics
    use precision_mod
    use energetics_mod, only : EnergeticsState, initialize_energetics, &
        update_energetics, finalize_energetics
    use theory_mod, only : compute_energetics_theory
    implicit none

    integer, parameter :: n = 3
    real(dp), parameter :: tolerance = 1.0e-13_dp
    real(dp), parameter :: dt = 0.2_dp
    integer :: i
    real(dp) :: noise(n, n), Q(n, n), S(n, n), A(n, n)
    real(dp) :: delta_x_old(n), delta_x_new(n)
    real(dp) :: delta_x_mid(n), d_delta_x(n)
    real(dp) :: force_c(n), force_nc(n)
    real(dp) :: expected_heat(n), expected_work(n)
    real(dp) :: expected_internal(n)
    real(dp) :: alpha(n, n)
    real(dp), allocatable :: heat_rate(:), work_rate(:)
    real(dp), allocatable :: internal_rate(:), entropy_rate(:)
    type(EnergeticsState) :: energy

    Q = reshape([ &
        -1.2_dp,  0.2_dp, -0.1_dp, &
         0.3_dp, -0.8_dp,  0.4_dp, &
        -0.2_dp,  0.1_dp, -1.5_dp  &
        ], [n, n])

    noise = 0.0_dp
    do i = 1, n
        noise(i, i) = 0.4_dp
    end do

    delta_x_old = [0.3_dp, -0.4_dp, 0.2_dp]
    delta_x_new = [0.32_dp, -0.35_dp, 0.18_dp]
    delta_x_mid = 0.5_dp * (delta_x_old + delta_x_new)
    d_delta_x = delta_x_new - delta_x_old

    S = 0.5_dp * (Q + transpose(Q))
    A = 0.5_dp * (Q - transpose(Q))
    force_c = matmul(S, delta_x_mid)
    force_nc = matmul(A, delta_x_mid)

    expected_heat = -(force_c + force_nc) * d_delta_x
    expected_work = -force_nc * d_delta_x
    expected_internal = -force_c * d_delta_x

    call initialize_energetics(energy, Q, noise)
    call update_energetics( &
        energy, delta_x_old, delta_x_new, dt)
    call finalize_energetics( &
        energy, heat_rate, work_rate, internal_rate, entropy_rate)

    if (maxval(abs(heat_rate * dt - expected_heat)) > tolerance) then
        error stop "Simulation heat does not use Q = S + A"
    end if
    if (maxval(abs(work_rate * dt - expected_work)) > tolerance) then
        error stop "Simulation work does not use antisymmetric A"
    end if
    if (maxval(abs(internal_rate * dt - expected_internal)) > tolerance) then
        error stop "Simulation internal energy does not use symmetric S"
    end if
    if (maxval(abs(expected_heat - expected_work - &
        expected_internal)) > tolerance) then
        error stop "Simulation per-step first law failed"
    end if

    alpha = reshape([ &
         0.0_dp, -0.4_dp,  0.2_dp, &
         0.4_dp,  0.0_dp, -0.3_dp, &
        -0.2_dp,  0.3_dp,  0.0_dp  &
        ], [n, n])

    call compute_energetics_theory( &
        Q, noise, alpha, heat_rate, work_rate, &
        internal_rate, entropy_rate)

    do i = 1, n
        if (abs(heat_rate(i) + &
            0.5_dp * dot_product(Q(i, :), alpha(:, i))) > tolerance) then
            error stop "Incorrect theoretical heat rate"
        end if
        if (abs(work_rate(i) + &
            0.5_dp * dot_product(A(i, :), alpha(:, i))) > tolerance) then
            error stop "Incorrect theoretical work rate"
        end if
        if (abs(internal_rate(i) + &
            0.5_dp * dot_product(S(i, :), alpha(:, i))) > tolerance) then
            error stop "Incorrect theoretical internal-energy rate"
        end if
    end do

    if (maxval(abs(heat_rate - work_rate - internal_rate)) > tolerance) then
        error stop "Theoretical rate decomposition failed"
    end if

    print *, "Unified linearized energetics tests passed."

end program test_linearized_energetics
