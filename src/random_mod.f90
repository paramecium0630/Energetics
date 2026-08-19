module random_mod
    use precision_mod
    implicit none

contains

    subroutine initialize_seed(seed)
        integer, intent(in) :: seed
        integer :: n, i
        integer, allocatable :: seed_values(:)
        ! Initialize the random number generator with a given seed

        call random_seed(size = n) ! Get the size of the seed array
        allocate(seed_values(n))

        do i = 1, n
            seed_values(i) = seed + i - 1
        end do
        
        call random_seed(put=seed_values)
        deallocate(seed_values)

    end subroutine initialize_seed

    real(dp) function rand_uniform() result(u)
        ! Generate a random number uniformly distributed in [0, 1)
        
        call random_number(u)

    end function rand_uniform

    real(dp) function rand_normal() result(z)
        ! Generate a random number from a standard normal distribution using Box-Muller transform
        real(dp) :: u1, u2
        real(dp) :: r, theta

        call random_number(u1)
        call random_number(u2)
        
        u1 = max(u1, tiny(1.0_dp))
        r = sqrt(-2.0_dp * log(u1))
        theta = 2.0_dp * acos(-1.0_dp) * u2

        z = r * cos(theta)

    end function rand_normal

end module random_mod