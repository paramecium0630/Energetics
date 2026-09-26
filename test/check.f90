program check
    use precision_mod
    use parameter_mod, only : SimulationParameters, initialize_bias
    implicit none

    type(SimulationParameters) :: param
    integer :: n_bias
    integer, allocatable :: bias_layer(:)
    real(dp), allocatable :: bias(:)
    character(len=16) :: resolved_mode

    param%N = 894
    param%graph_type = "EXTERNAL"
    param%bias_mode = "AUTO"
    param%bias_file = "input/mnist100x1_self/bias.dat"

    call initialize_bias(param, bias, bias_layer, n_bias, resolved_mode)

    if (trim(resolved_mode) /= "FILE") then
        error stop "AUTO did not select FILE for an external network"
    end if

    if (size(bias) /= 894) then
        error stop "Incorrect bias-array size"
    end if

    if (n_bias /= 110) then
        error stop "Incorrect number of biases"
    end if

    if (count(bias_layer == 0) /= 784) then
        error stop "Incorrect number of no-bias nodes"
    end if

    if (count(bias_layer == 2) /= 100) then
        error stop "Incorrect layer-2 count"
    end if

    if (count(bias_layer == 3) /= 10) then
        error stop "Incorrect layer-3 count"
    end if

    if (abs(bias(785) - 4.5183491893112659e-03_dp) > 1.0e-14_dp) then
        error stop "Incorrect first layer-2 bias"
    end if

    if (abs(bias(894) + 7.3095090687274933e-02_dp) > 1.0e-14_dp) then
        error stop "Incorrect final output bias"
    end if

    print *, "Bias reader test passed."

end program check
