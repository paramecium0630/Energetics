program check
    use precision_mod
    use parameter_mod, only : read_node_bias
    implicit none

    integer :: n_bias
    integer, allocatable :: bias_layer(:)
    real(dp), allocatable :: bias(:)

    call read_node_bias( &
        "input/bias.dat", &
        1306, &
        bias, &
        bias_layer, &
        n_bias)

    if (size(bias) /= 1306) then
        error stop "Incorrect bias-array size"
    end if

    if (n_bias /= 522) then
        error stop "Incorrect number of biases"
    end if

    if (count(bias_layer == 0) /= 784) then
        error stop "Incorrect number of no-bias nodes"
    end if

    if (count(bias_layer == 1) /= 256) then
        error stop "Incorrect layer-1 count"
    end if

    if (count(bias_layer == 2) /= 256) then
        error stop "Incorrect layer-2 count"
    end if

    if (count(bias_layer == 3) /= 10) then
        error stop "Incorrect layer-3 count"
    end if

    if (abs(bias(785) - 5.0167959183454514e-02_dp) > 1.0e-14_dp) then
        error stop "Incorrect first layer-1 bias"
    end if

    if (abs(bias(1306) - 9.9143639206886292e-02_dp) > 1.0e-14_dp) then
        error stop "Incorrect final output bias"
    end if

    print *, "Bias reader test passed."

end program check