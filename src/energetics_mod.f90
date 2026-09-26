module energetics_mod
    use precision_mod
    implicit none

    type, public :: EnergeticsState
        integer :: n_step = 0
        real(dp) :: elapsed_time = 0.0_dp

        real(dp), allocatable :: S(:,:)
        real(dp), allocatable :: A(:,:)
        real(dp), allocatable :: sigma_diag(:)

        real(dp), allocatable :: sum_heat(:)
        real(dp), allocatable :: sum_work(:)
        real(dp), allocatable :: sum_internal(:)

        ! Reusable workspace for each sampling step
        real(dp), allocatable :: d_delta_x(:)
        real(dp), allocatable :: delta_x_mid(:)
        real(dp), allocatable :: force_c(:)
        real(dp), allocatable :: force_nc(:)

    end type EnergeticsState

    public :: initialize_energetics
    public :: update_energetics
    public :: finalize_energetics
    public :: aggregate_layer_energetics

contains

    subroutine initialize_energetics(energy, Q, noise)
        type(EnergeticsState), intent(out) :: energy
        integer :: n, i
        real(dp), intent(in) :: Q(:,:), noise(:,:)

        n = size(Q, 1)

        allocate(energy%S(n,n), energy%A(n,n), energy%sigma_diag(n))

        allocate(energy%sum_heat(n))
        allocate(energy%sum_work(n))
        allocate(energy%sum_internal(n))

        allocate(energy%d_delta_x(n))
        allocate(energy%delta_x_mid(n))
        allocate(energy%force_c(n))
        allocate(energy%force_nc(n))

        if (size(Q,2) /= n) error stop "Q must be square"

        if (size(noise,1) /= n .or. size(noise,2) /= n) then
            error stop "Q and noise size mismatch"
        end if

        ! decompose force
        energy%S = 0.5_dp * (Q + transpose(Q))
        energy%A = 0.5_dp * (Q - transpose(Q))

        ! noise diagonal
        do i = 1, n
            if (noise(i,i) <= 0.0_dp) then
                error stop "Noise diagonal must be positive"
            end if

            energy%sigma_diag(i) = noise(i,i)
        end do

        energy%sum_heat     = 0.0_dp
        energy%sum_work     = 0.0_dp
        energy%sum_internal = 0.0_dp
        energy%d_delta_x    = 0.0_dp
        energy%delta_x_mid  = 0.0_dp
        energy%force_c      = 0.0_dp
        energy%force_nc     = 0.0_dp
        energy%n_step       = 0
        energy%elapsed_time = 0.0_dp

    end subroutine initialize_energetics

    subroutine update_energetics(energy, delta_x_old, delta_x_new, dt)
        ! Linearized energetics about the fixed point for every coupling type.
        ! The force Jacobian is decomposed as Q = S + A, where S is
        ! symmetric (conservative) and A is antisymmetric (nonconservative).
        type(EnergeticsState), intent(inout) :: energy
        integer :: n, i
        real(dp), intent(in) :: delta_x_old(:), delta_x_new(:)
        real(dp), intent(in) :: dt

        n = size(delta_x_old)

        if (size(delta_x_new) /= n) then
            error stop "delta_x_old and delta_x_new size mismatch"
        end if
        if (.not. allocated(energy%d_delta_x)) then
            error stop "Energetics workspace is not initialized"
        end if
        if (size(energy%d_delta_x) /= n) then
            error stop "Energetics workspace size mismatch"
        end if
        if (dt <= 0.0_dp) error stop "dt must be positive"

        energy%d_delta_x   = delta_x_new - delta_x_old
        energy%delta_x_mid = 0.5_dp * (delta_x_old + delta_x_new)

        ! conservative and nonconservative force
        energy%force_c  = matmul(energy%S, energy%delta_x_mid)
        energy%force_nc = matmul(energy%A, energy%delta_x_mid)

        do i = 1, n
            energy%sum_heat(i) = energy%sum_heat(i) - &
                (energy%force_c(i) + energy%force_nc(i)) * energy%d_delta_x(i)
            energy%sum_work(i) = energy%sum_work(i) - &
                energy%force_nc(i) * energy%d_delta_x(i)
            energy%sum_internal(i) = energy%sum_internal(i) - &
                energy%force_c(i) * energy%d_delta_x(i)
        end do

        energy%n_step       = energy%n_step + 1
        energy%elapsed_time = energy%elapsed_time + dt

    end subroutine update_energetics

    subroutine finalize_energetics(energy, heat_rate, work_rate, &
                               internal_rate, entropy_rate)
        type(EnergeticsState), intent(inout) :: energy

        real(dp), allocatable, intent(out) :: heat_rate(:)
        real(dp), allocatable, intent(out) :: work_rate(:)
        real(dp), allocatable, intent(out) :: internal_rate(:)
        real(dp), allocatable, intent(out) :: entropy_rate(:)

        heat_rate     = energy%sum_heat     / energy%elapsed_time
        work_rate     = energy%sum_work     / energy%elapsed_time
        internal_rate = energy%sum_internal / energy%elapsed_time

        entropy_rate = -2.0_dp * heat_rate / energy%sigma_diag

        if (allocated(energy%S)) deallocate(energy%S)
        if (allocated(energy%A)) deallocate(energy%A)
        if (allocated(energy%sigma_diag)) deallocate(energy%sigma_diag)
        if (allocated(energy%sum_heat)) deallocate(energy%sum_heat)
        if (allocated(energy%sum_work)) deallocate(energy%sum_work)
        if (allocated(energy%sum_internal)) deallocate(energy%sum_internal)
        if (allocated(energy%d_delta_x)) deallocate(energy%d_delta_x)
        if (allocated(energy%delta_x_mid)) deallocate(energy%delta_x_mid)
        if (allocated(energy%force_c)) deallocate(energy%force_c)
        if (allocated(energy%force_nc)) deallocate(energy%force_nc)

        energy%n_step = 0
        energy%elapsed_time = 0.0_dp

    end subroutine finalize_energetics


    subroutine aggregate_layer_energetics( &
        node_layer, heat_rate, work_rate, internal_rate, entropy_rate, &
        node_count, layer_heat, layer_work, layer_internal, layer_entropy)
        ! 單次掃描節點，將節點速率加總成各層總速率；不要求節點按層排列。
        ! 呼叫端只在已確認 FCNN 且建立 node_layer 後使用此函式。
        integer, intent(in) :: node_layer(:)
        real(dp), intent(in) :: heat_rate(:), work_rate(:)
        real(dp), intent(in) :: internal_rate(:), entropy_rate(:)
        integer, allocatable, intent(out) :: node_count(:)
        real(dp), allocatable, intent(out) :: layer_heat(:), layer_work(:)
        real(dp), allocatable, intent(out) :: layer_internal(:), layer_entropy(:)
        integer :: n, n_layers, node, layer

        n = size(node_layer)
        if (n <= 0) error stop "Layer aggregation requires at least one node"
        if (size(heat_rate) /= n .or. size(work_rate) /= n .or. &
            size(internal_rate) /= n .or. size(entropy_rate) /= n) then
            error stop "Layer IDs and energetic rates size mismatch"
        end if
        if (any(node_layer < 1) .or. any(node_layer > n)) then
            error stop "Layer IDs must be between 1 and the node count"
        end if
        n_layers = maxval(node_layer)
        allocate(node_count(n_layers))
        allocate(layer_heat(n_layers), layer_work(n_layers))
        allocate(layer_internal(n_layers), layer_entropy(n_layers))
        node_count = 0
        layer_heat = 0.0_dp
        layer_work = 0.0_dp
        layer_internal = 0.0_dp
        layer_entropy = 0.0_dp

        do node = 1, n
            layer = node_layer(node)
            node_count(layer) = node_count(layer) + 1
            layer_heat(layer) = layer_heat(layer) + heat_rate(node)
            layer_work(layer) = layer_work(layer) + work_rate(node)
            layer_internal(layer) = layer_internal(layer) + internal_rate(node)
            ! 必須加總節點 EPR，不能用層平均噪音除層總熱率。
            layer_entropy(layer) = layer_entropy(layer) + entropy_rate(node)
        end do
        if (any(node_count == 0)) then
            error stop "Layer IDs must be consecutive with no empty layers"
        end if
    end subroutine aggregate_layer_energetics

end module energetics_mod
