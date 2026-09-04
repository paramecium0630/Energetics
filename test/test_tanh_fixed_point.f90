program test_tanh_fixed_point
    use precision_mod
    use langevin_mod, only : compute_force, construct_Q
    use theory_mod, only : solve_fixed_point_tanh
    implicit none

    integer, parameter :: n = 3
    real(dp), parameter :: finite_difference_step = 1.0e-6_dp
    real(dp), parameter :: jacobian_tolerance = 1.0e-9_dp
    real(dp), parameter :: fixedpoint_tolerance = 1.0e-10_dp

    integer :: i, j
    real(dp) :: r(n), W(n, n), bias(n)
    real(dp) :: expected_fixpoint(n), computed_fixpoint(n)
    real(dp) :: force(n), force_plus(n), force_minus(n)
    real(dp) :: state_plus(n), state_minus(n)
    real(dp) :: finite_difference_column(n)
    real(dp) :: expected_Q(n, n)
    real(dp), allocatable :: Q(:, :)

    r = [2.0_dp, 1.5_dp, 2.5_dp]

    W = 0.0_dp
    W(1, 2) =  0.30_dp
    W(1, 3) = -0.20_dp
    W(2, 1) =  0.10_dp
    W(2, 3) =  0.25_dp
    W(3, 1) = -0.15_dp
    W(3, 2) =  0.20_dp

    expected_fixpoint = [0.20_dp, -0.15_dp, 0.10_dp]

    ! Choose bias so that expected_fixpoint is an exact TANH fixed point.
    bias = r * expected_fixpoint - matmul(W, tanh(expected_fixpoint))

    call solve_fixed_point_tanh( &
        r, W, bias, computed_fixpoint, &
        1.0e-12_dp, 100)

    if (maxval(abs(computed_fixpoint - expected_fixpoint)) > &
        fixedpoint_tolerance) then
        error stop "TANH solver returned an incorrect fixed point"
    end if

    call compute_force( &
        computed_fixpoint, r, W, bias, "TANH", force)

    if (maxval(abs(force)) > 1.0e-12_dp) then
        error stop "TANH fixed-point residual is too large"
    end if

    call construct_Q( &
        r, W, "TANH", Q, expected_fixpoint)

    ! Compare every Jacobian column with a centered finite difference.
    do j = 1, n
        state_plus = expected_fixpoint
        state_minus = expected_fixpoint
        state_plus(j) = state_plus(j) + finite_difference_step
        state_minus(j) = state_minus(j) - finite_difference_step

        call compute_force( &
            state_plus, r, W, bias, "TANH", force_plus)
        call compute_force( &
            state_minus, r, W, bias, "TANH", force_minus)

        finite_difference_column = &
            (force_plus - force_minus) / &
            (2.0_dp * finite_difference_step)

        if (maxval(abs(Q(:, j) - finite_difference_column)) > &
            jacobian_tolerance) then
            error stop "TANH Jacobian failed finite-difference test"
        end if
    end do

    ! Exercise the triangular Newton path used by feed-forward networks.
    W = 0.0_dp
    W(2, 1) =  0.40_dp
    W(3, 1) = -0.20_dp
    W(3, 2) =  0.30_dp

    expected_fixpoint = [0.10_dp, -0.20_dp, 0.15_dp]
    bias = r * expected_fixpoint - matmul(W, tanh(expected_fixpoint))

    call solve_fixed_point_tanh( &
        r, W, bias, computed_fixpoint, &
        1.0e-12_dp, 100)

    if (maxval(abs(computed_fixpoint - expected_fixpoint)) > &
        fixedpoint_tolerance) then
        error stop "Triangular TANH Newton solve failed"
    end if

    call compute_force( &
        computed_fixpoint, r, W, bias, "TANH", force)

    if (maxval(abs(force)) > 1.0e-12_dp) then
        error stop "Triangular TANH residual is too large"
    end if

    ! Check that the original DIFFUSIVE Q formula is unchanged.
    call construct_Q(r, W, "DIFFUSIVE", Q)

    expected_Q = W
    do i = 1, n
        expected_Q(i, i) = -r(i) - sum(W(i, :)) + W(i, i)
    end do

    if (maxval(abs(Q - expected_Q)) > 0.0_dp) then
        error stop "DIFFUSIVE Q changed after coupling dispatch was added"
    end if

    print *, "TANH fixed-point and Jacobian tests passed."

end program test_tanh_fixed_point
