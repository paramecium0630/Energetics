program test_random_bias
    use precision_mod
    use parameter_mod, only : SimulationParameters, set_random_bias
    use random_mod, only : initialize_seed
    implicit none

    type(SimulationParameters) :: param
    real(dp), allocatable :: bias_first(:), bias_second(:)

    param%N = 8
    param%bias_mean = 0.5_dp
    param%bias_std = 0.2_dp

    call initialize_seed(1234)
    call set_random_bias(param, bias_first)

    call initialize_seed(1234)
    call set_random_bias(param, bias_second)

    if (size(bias_first) /= param%N) then
        error stop "Incorrect random-bias size"
    end if

    if (maxval(abs(bias_first - bias_second)) > 0.0_dp) then
        error stop "Random bias is not reproducible for a fixed seed"
    end if

    if (all(abs(bias_first - param%bias_mean) <= tiny(1.0_dp))) then
        error stop "Random bias has no variation"
    end if

    print *, "Random-bias test passed."

end program test_random_bias
