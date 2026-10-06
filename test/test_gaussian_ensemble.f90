program test_gaussian_ensemble
    use precision_mod
    use parameter_mod
    use random_mod
    use network_mod
    use gaussian_mod
    implicit none
    type(SimulationParameters) :: p
    logical, allocatable :: adj(:,:)
    real(dp), allocatable :: w(:,:), saved(:,:)
    real(dp) :: r(6), noise(6,6), bias(6), expected, total, norm, residual, eig
    real(dp) :: means(2), stds(2), values(3), repeat_values(3)
    integer :: layers(6), i, unit, id, pass
    character(len=256) :: header
    character(len=16) :: status
    p%N=6
    p%directed=.true.
    p%n__samples=3
    p%coupling_type="LINEAR"
    p%verify_lyapunov=.true.
    p%seed=42
    layers=[1,1,2,2,3,3]
    means=[0.1_dp,-0.1_dp]
    stds=[0.2_dp,0.3_dp]
    r=10
    noise=0
    do i=1,6
        noise(i,i)=1
    end do
    bias=0
    call execute_command_line('mkdir -p build/test_gaussian_ensemble')
    do pass=1,2
        call initialize_seed(p%seed)
        call generate_fcnn(p,1,[2,2,2],means,stds,adj,w)
        expected=sum(w**2)/20 + sum(matmul(w(5:6,3:4),w(3:4,1:2))**2)/8000
        call run_gaussian_ensemble(p,[2,2,2],means,stds,layers,r,noise,bias,adj,w, &
            "build/test_gaussian_ensemble")
        open(newunit=unit,file="build/test_gaussian_ensemble/gaussian_trials.csv",status="old")
        read(unit,'(A)') header
        do i=1,3
            read(unit,*) id,status,eig,total,norm,residual
            if (id /= i .or. trim(status) /= "stable") error stop "Invalid Gaussian sample"
            if (residual > 1e-12_dp) error stop "Gaussian covariance residual"
            if (i == 1 .and. abs(total-expected) > 1e-12_dp) error stop "First Gaussian draw lost"
            if (pass == 1) then
                values(i)=total
            else
                repeat_values(i)=total
            end if
        end do
        close(unit)
        if (pass == 1) saved=w
    end do
    if (any(values /= repeat_values) .or. any(saved /= w)) error stop "Gaussian reproducibility"
    if (values(1) == values(2) .or. values(2) == values(3)) error stop "Repeated Gaussian draws"
    print *, "Gaussian ensemble tests passed"
end program test_gaussian_ensemble
