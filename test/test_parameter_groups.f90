program test_parameter_groups
    use precision_mod
    use parameter_mod, only : SimulationParameters, read_parameters
    implicit none

    type(SimulationParameters) :: param

    call read_parameters("test/parameters_dynamics.nml", param)

    if (abs(param%sigma_mean - 0.321_dp) > epsilon(1.0_dp)) then
        error stop "sigma_mean was not read from &dynamics"
    end if
    if (trim(param%coupling_type) /= "TANH") then
        error stop "Other dynamics parameters changed while reading sigma_mean"
    end if
    if (param%N /= 4 .or. param%seed /= 99) then
        error stop "Namelist groups after the dynamics merge were misread"
    end if
    if (trim(param%shuffle_mode) /= "BOTH") then
        error stop "shuffle_mode was not read from &theory"
    end if

    print *, "Dynamics sigma namelist test passed."

end program test_parameter_groups
