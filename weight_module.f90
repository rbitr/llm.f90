! set precision of reals (here default which is 4 byte)
module precision_module
  implicit none
  integer, parameter :: wp = kind(1.0)
end module precision_module

! structs for reading weights, config information and state 
module weight_module
        use precision_module
        implicit none
        private wp

        type TransformerWeights
                real(kind=wp), allocatable :: token_embedding_table(:,:)
                real(kind=wp), allocatable :: rms_att_weight(:,:)
                real(kind=wp), allocatable :: rms_ffn_weight(:,:)
                real(kind=wp), allocatable :: wqkv(:,:,:)
                real(kind=wp), allocatable :: wo(:,:,:)
                real(kind=wp), allocatable :: w13(:,:,:)
                real(kind=wp), allocatable :: w2(:,:,:)
                real(kind=wp), allocatable :: rms_final_weight(:)
                !real(kind=wp), allocatable :: freq_cis_real(:,:)
                !real(kind=wp), allocatable :: freq_cis_imag(:,:)
                real(kind=wp), allocatable :: wcls(:,:)

        end type TransformerWeights

        type Config
                integer :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
                integer :: kv_head_size
        end type Config

        type RunState

                real(kind=wp), allocatable :: att(:,:)
                real(kind=wp), allocatable :: key_cache(:,:,:)
                real(kind=wp), allocatable :: value_cache(:,:,:)
                real(kind=wp) :: times(5)

        end type RunState

end module weight_module

