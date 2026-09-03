program test_fixed_point
    use precision_mod
    use theory_mod, only : solve_fixed_point_linear
    implicit none

    real(dp) :: Q(2, 2)
    real(dp) :: bias(2)
    real(dp) :: fixpoint(2)
    real(dp) :: residual

    Q = reshape([ &
        -2.0_dp, 1.0_dp, &
         0.0_dp, -3.0_dp], [2, 2])

    bias = [2.0_dp, 5.0_dp]

    call solve_fixed_point_linear( &
        Q, bias, &
        .false., .true., &
        fixpoint)

    if (maxval(abs(fixpoint - [1.0_dp, 2.0_dp])) > 1.0e-12_dp) then
        error stop "Incorrect triangular fixed point"
    end if

    residual = maxval(abs(matmul(Q, fixpoint) + bias))

    if (residual > 1.0e-12_dp) then
        error stop "Fixed-point residual is too large"
    end if

    print *, "Fixed-point test passed."

end program test_fixed_point