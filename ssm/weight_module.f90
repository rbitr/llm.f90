! set precision of reals (here default which is 4 byte)
module precision_module
  implicit none
  integer, parameter :: wp = kind(1.0)
  integer, parameter :: f16p = 2
end module precision_module

! structs for reading weights, config information and state 
module weight_module
        use precision_module
        implicit none
        private wp

        type fp321d
                real(kind=wp), allocatable :: m(:)
        end type

        type fp322d
                real(kind=wp), allocatable :: m(:,:)
        end type

        type fp323d
                real(kind=wp), allocatable :: m(:,:,:)
        end type

        type MambaWeights
                real(kind=wp), allocatable :: token_embedding_table(:,:)
                type(fp321d), allocatable :: D(:)
                type(fp322d), allocatable :: in_proj_weight(:)
                type(fp323d), allocatable :: conv1d_weight(:)
                type(fp321d), allocatable :: conv1d_bias(:)
                type(fp322d), allocatable :: x_proj_weight(:)
                type(fp322d), allocatable :: dt_proj_weight(:)
                type(fp321d), allocatable :: dt_proj_bias(:)
                type(fp322d), allocatable :: A_log(:)
                type(fp322d), allocatable :: out_proj_weight(:)
                type(fp321d), allocatable :: norm_weight(:)
                real(kind=wp), allocatable :: norm_f_weight(:)
                real(kind=wp), allocatable :: wcls(:,:)
        end type MambaWeights

        type Config
                integer :: d_model, n_layer, vocab_size
        end type Config

        ! update for internal states
        type RunState

                type(fp322d), allocatable :: conv_state(:)
                type(fp322d), allocatable :: ssm_state(:)
                integer :: seqlen_offset
                real(kind=wp) :: times(5)

        end type RunState

end module weight_module

