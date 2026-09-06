program test_tanh_energetics
    use precision_mod
    use energetics_mod, only : EnergeticsState, initialize_energetics, &
        update_energetics_linear, update_energetics_tanh, &
        finalize_energetics
    use langevin_mod, only : compute_force
    use theory_mod, only : compute_energetics_theory_tanh
    implicit none

    integer, parameter :: n = 3
    real(dp), parameter :: tolerance = 1.0e-13_dp
    real(dp), parameter :: dt = 0.2_dp
    integer :: i
    real(dp) :: r(n), W(n, n), bias(n), noise(n, n), Q(n, n)
    real(dp) :: x_old(n), x_new(n), x_mid(n), delta_x(n)
    real(dp) :: force_c(n), force_nc(n), force_total(n)
    real(dp) :: expected_heat(n)
    real(dp) :: expected_work(n), expected_internal(n)
    real(dp) :: S(n, n), A(n, n), delta_x_mid(n)
    real(dp) :: potential_old(n), potential_new(n)
    real(dp) :: alpha(n, n), B(n, n)
    real(dp), allocatable :: heat_rate(:), work_rate(:)
    real(dp), allocatable :: internal_rate(:), entropy_rate(:)
    type(EnergeticsState) :: energy

    r = [1.2_dp, 0.8_dp, 1.5_dp]
    bias = [0.1_dp, -0.2_dp, 0.05_dp]
    x_old = [0.3_dp, -0.4_dp, 0.2_dp]
    x_new = [0.32_dp, -0.35_dp, 0.18_dp]

    W = reshape([ &
         0.0_dp,  0.2_dp, -0.1_dp, &
         0.3_dp,  0.0_dp,  0.4_dp, &
        -0.2_dp,  0.1_dp,  0.0_dp  &
        ], [n, n])

    noise = 0.0_dp
    do i = 1, n
        noise(i, i) = 0.4_dp
    end do
    Q = -0.5_dp
    do i = 1, n
        Q(i, i) = -r(i)
    end do

    x_mid = 0.5_dp * (x_old + x_new)
    delta_x = x_new - x_old
    force_c = -r * x_mid + bias
    force_nc = matmul(W, tanh(x_mid))
    call compute_force(x_mid, r, W, bias, "TANH", force_total)

    if (maxval(abs(force_total - force_c - force_nc)) > tolerance) then
        error stop "TANH energetic-force split differs from dynamics"
    end if

    expected_heat = -(force_c + force_nc) * delta_x
    expected_work = -force_nc * delta_x
    expected_internal = -force_c * delta_x

    call initialize_energetics(energy, Q, noise)
    call update_energetics_tanh( &
        energy, x_old, x_new, r, W, bias, dt)
    call finalize_energetics( &
        energy, heat_rate, work_rate, internal_rate, entropy_rate)

    if (maxval(abs(heat_rate * dt - expected_heat)) > tolerance) then
        error stop "TANH heat does not use the full midpoint force"
    end if
    if (maxval(abs(work_rate * dt - expected_work)) > tolerance) then
        error stop "TANH work does not use the nonlinear coupling force"
    end if
    if (maxval(abs(internal_rate * dt - expected_internal)) > tolerance) then
        error stop "TANH internal energy uses an incorrect force"
    end if
    if (maxval(abs(expected_heat - expected_work - &
        expected_internal)) > tolerance) then
        error stop "TANH per-step first law failed"
    end if

    potential_old = 0.5_dp * r * x_old**2 - bias * x_old
    potential_new = 0.5_dp * r * x_new**2 - bias * x_new
    if (maxval(abs(expected_internal - &
        (potential_new - potential_old))) > tolerance) then
        error stop "TANH conservative midpoint energy identity failed"
    end if

    alpha = reshape([ &
         0.0_dp, -0.4_dp,  0.2_dp, &
         0.4_dp,  0.0_dp, -0.3_dp, &
        -0.2_dp,  0.3_dp,  0.0_dp  &
        ], [n, n])
    B = Q
    do i = 1, n
        B(i, i) = B(i, i) + r(i)
    end do

    call compute_energetics_theory_tanh( &
        Q, r, noise, alpha, heat_rate, work_rate, &
        internal_rate, entropy_rate)

    do i = 1, n
        if (abs(heat_rate(i) + &
            0.5_dp * dot_product(Q(i, :), alpha(:, i))) > tolerance) then
            error stop "Incorrect TANH theoretical heat rate"
        end if
        if (abs(work_rate(i) + &
            0.5_dp * dot_product(B(i, :), alpha(:, i))) > tolerance) then
            error stop "Incorrect TANH theoretical work rate"
        end if
    end do
    if (maxval(abs(internal_rate)) > tolerance) then
        error stop "TANH theoretical internal rate should be zero"
    end if
    if (maxval(abs(heat_rate - work_rate)) > tolerance) then
        error stop "TANH steady theoretical heat and work should agree"
    end if

    ! Regression check: the existing DIFFUSIVE S/A calculation is unchanged.
    S = 0.5_dp * (Q + transpose(Q))
    A = 0.5_dp * (Q - transpose(Q))
    delta_x_mid = 0.5_dp * (x_old + x_new)
    force_c = matmul(S, delta_x_mid)
    force_nc = matmul(A, delta_x_mid)
    expected_heat = -(force_c + force_nc) * delta_x
    expected_work = -force_nc * delta_x
    expected_internal = -force_c * delta_x

    call initialize_energetics(energy, Q, noise)
    call update_energetics_linear(energy, x_old, x_new, dt)
    call finalize_energetics( &
        energy, heat_rate, work_rate, internal_rate, entropy_rate)

    if (maxval(abs(heat_rate * dt - expected_heat)) > tolerance .or. &
        maxval(abs(work_rate * dt - expected_work)) > tolerance .or. &
        maxval(abs(internal_rate * dt - expected_internal)) > tolerance) then
        error stop "DIFFUSIVE linear energetics regression failed"
    end if

    print *, "TANH energetics tests passed."

end program test_tanh_energetics
