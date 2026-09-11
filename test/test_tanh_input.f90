program test_tanh_input
    use precision_mod
    use langevin_mod, only : compute_force, construct_Q
    use theory_mod, only : solve_fixed_point_nonlinear
    implicit none
    real(dp) :: r(3), W(3,3), bias(3), expected(3), fixedpoint(3)
    real(dp) :: x(3), plus(3), minus(3), fp(3), fm(3), force(3)
    real(dp), allocatable :: Q(:,:)
    integer :: i
    real(dp), parameter :: h=1.0e-6_dp
    r = [1.5_dp, 2.0_dp, 2.5_dp]
    bias = [0.7_dp, -0.4_dp, 1.2_dp]
    W = 0.0_dp
    call solve_fixed_point_nonlinear(r,W,bias,"TANH_INPUT",fixedpoint,1.0e-12_dp,100)
    if (maxval(abs(fixedpoint-tanh(bias)/r)) > 1.0e-12_dp) error stop "Uncoupled TANH_INPUT"

    W(2,1)=0.8_dp
    W(3,1)=-0.3_dp
    W(3,2)=0.6_dp
    expected(1)=tanh(bias(1))/r(1)
    expected(2)=tanh(W(2,1)*expected(1)+bias(2))/r(2)
    expected(3)=tanh(dot_product(W(3,1:2),expected(1:2))+bias(3))/r(3)
    call solve_fixed_point_nonlinear(r,W,bias,"TANH_INPUT",fixedpoint,1.0e-12_dp,100)
    if (maxval(abs(fixedpoint-expected)) > 1.0e-11_dp) error stop "Feedforward fixed point"
    call compute_force(fixedpoint,r,W,bias,"TANH_INPUT",force)
    if (maxval(abs(force)) > 1.0e-12_dp) error stop "Fixed-point residual"

    ! Nontriangular, unequal inputs detect accidental column scaling.
    W(1,3)=0.2_dp
    x=[0.1_dp, -0.3_dp, 0.4_dp]
    call construct_Q(r,W,"TANH_INPUT",Q,x,bias)
    do i=1,3
        plus=x
        minus=x
        plus(i)=plus(i)+h
        minus(i)=minus(i)-h
        call compute_force(plus,r,W,bias,"TANH_INPUT",fp)
        call compute_force(minus,r,W,bias,"TANH_INPUT",fm)
        if (maxval(abs(Q(:,i)-(fp-fm)/(2*h))) > 1.0e-9_dp) error stop "Jacobian finite difference"
        if (abs(Q(i,i)+r(i)) > 1.0e-12_dp) error stop "Jacobian diagonal"
    end do
    call solve_fixed_point_nonlinear(r,W,bias,"TANH_INPUT",fixedpoint,1.0e-12_dp,100)
    call compute_force(fixedpoint,r,W,bias,"TANH_INPUT",force)
    if (maxval(abs(force)) > 1.0e-12_dp) error stop "General Newton residual"
    print *, "TANH_INPUT tests passed."
end program test_tanh_input
