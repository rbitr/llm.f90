! mamba.f90
! Andrew Marble
! http://marble.onl
! See license and notices


module arg_parse
        implicit none

        type args
                real :: temperature
                character(:), allocatable :: model_file
                character(:), allocatable :: prompt
                character(:), allocatable :: tokenizer
                logical :: verbose, ak
                integer :: n
        end type args

        contains

                subroutine parse_args(arg_values)
                        type(args) :: arg_values 
                        integer :: i, num_args
                        character(256) :: arg

                        

                        !defaults 
                        arg_values%temperature = 0
                        arg_values%model_file = ""
                        arg_values%prompt = ""
                        arg_values%verbose = .false.
                        arg_values%n = 256
                        arg_values%tokenizer = ""
                        arg_values%ak = .false.
                
                        num_args = command_argument_count()

                        i = 1
                        do while (i <= num_args)
                                call get_command_argument(i, arg)
                                        select case (arg)
                                                case ('-m', '--model')
                                                ! path to model file
                                                call get_command_argument(i+1, arg)
                                                arg_values%model_file = trim(arg)
                                                i = i + 2
                                                case ('-p', '--prompt')
                                                ! prompt string
                                                call get_command_argument(i+1, arg)
                                                arg_values%prompt = trim(arg)
                                                i = i + 2
                                                case ('-s', '--tokenizer')
                                                ! path to custom tokenizer
                                                call get_command_argument(i+1, arg)
                                                arg_values%tokenizer = trim(arg)
                                                i = i + 2
                                                case ('-t', '--temperature')
                                                ! temperature scaling
                                                call get_command_argument(i+1, arg)
                                                read(arg,*) arg_values%temperature
                                                i = i + 2
                                                case ('-n', '--num_tokens')
                                                ! number of tokens to generate, including prompt
                                                call get_command_argument(i+1, arg)
                                                read(arg,*) arg_values%n
                                                i = i + 2
                                                case ('-v', '--verbose')
                                                ! print additional information
                                                arg_values%verbose = .true.
                                                i = i + 1
                                                case ('--ak')
                                                ! llama2.c file format
                                                arg_values%ak = .true.
                                                i = i + 1
                                                case default
                                                print *, 'Unrecognized option:', trim(arg)
                                                stop
                                                end select
                        end do



                end subroutine

end module arg_parse



program ssm_inference
        use, intrinsic :: iso_fortran_env, only : iostat_end
        use iso_c_binding
        use precision_module
        use weight_module
        use arg_parse
        !use read_ggml, only: load_ggml
        !use omp_lib

        implicit none


        
        ! weights and states
        integer :: dummy(7)
        integer :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        integer, parameter :: d_conv = 4
        integer, parameter :: expand = 2
        integer, parameter :: d_state = 16
        integer :: d_inner, dt_rank, d_model

        !integer, parameter :: emb_dim = 2048
        !integer, parameter :: hidden_dim = 5632
        !integer, parameter :: n_layers = 22
        !integer, parameter :: n_heads = 32
        !integer, parameter :: n_kv_heads = 4
        !integer, parameter :: vocab_size = 32000
        !integer :: seq_len = 2048

        type(MambaWeights) :: weights
        logical :: shared_weights
        integer :: head_size, kv_head_size, tmp
        type(config) :: conf, dummy_conf
        type(RunState) :: s
        real(kind=wp), allocatable :: logits(:)
        real(kind=wp), allocatable :: freq_buf(:)
        real(kind=wp), allocatable :: temp1(:), temp2(:,:), temp3(:,:,:)
        integer :: l

        !for the tokens

        integer :: pos
        integer :: token        
        real(kind=wp) :: score
        integer :: tok_len, max_len, n
        !integer :: vocab_size = 32000
        character(:), allocatable :: tmpstr
        character(:), dimension(:), allocatable :: vocab
        real(kind=wp),allocatable :: scores(:)
        integer, allocatable :: prompt_tokens(:)
        integer, allocatable :: vocab_len(:)
        
        ! command line arguments
        !integer :: num_args
        !character(64) :: arg
        type (args) :: arg_values
        real :: temperature
        character(:), allocatable :: prompt
        logical :: verbose

        ! timing
        real(kind=wp) :: t_ms_start, t_ms_end
        
        integer :: error

        call parse_args(arg_values)

        verbose = arg_values%verbose
        
        if (.not. arg_values%ak) then

        print *, "Only packed weights format currently implemented, please use the --ak option"
        stop
        !call load_ggml(arg_values%model_file, weights, dummy_conf, vocab, scores, vocab_len, verbose)
        !max_len = maxval(vocab_len)-2
        !head_size = emb_dim / n_heads
        !kv_head_size = n_kv_heads * head_size
        !shared_weights =.false.


        ! open the model file 
        else
        open(UNIT=5, FILE=arg_values%model_file, FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
                ! config
                read(5) emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
                !read(5) dummy
                d_model = emb_dim ! clean this up ie only one name for emb_dim and d_model
                d_inner = expand * emb_dim
                dt_rank = ceiling( emb_dim / 16.0 )
                !head_size = emb_dim / n_heads
                !kv_head_size = n_kv_heads * head_size
                
                if (verbose) then
                        print *, "Embedding dimension: (d_model)", emb_dim
                        print *, "Layers: ", n_layers
                        print *, "Vocabulary Size: ", vocab_size
                        print *, "d_inner", d_inner
                        print *, "dt_rank", dt_rank

                end if 

                
                allocate(weights%token_embedding_table(emb_dim,vocab_size))
                read(5) weights%token_embedding_table
                
                if (verbose) then
                        print *, "loaded embedding weights:", size(weights%token_embedding_table)
                end if 

                allocate(weights%D(n_layers))
                do l = 1,n_layers
                allocate(weights%D(l)%m(d_inner))
                read(5) weights%D(l)%m
                end do

                if (verbose) then
                        print *, "loaded D weights:", n_layers, shape(weights%D(1)%m)
                end if

                allocate(weights%in_proj_weight(n_layers))
                do l = 1,n_layers
                allocate(weights%in_proj_weight(l)%m(d_model,2*d_inner))
                read(5) weights%in_proj_weight(l)%m
                end do

                if (verbose) then
                        print *, "loaded in projection weights:", n_layers, shape(weights%in_proj_weight(1)%m)
                end if

                allocate(weights%conv1d_weight(n_layers))
                do l = 1,n_layers
                allocate(weights%conv1d_weight(l)%m(d_conv,1,d_inner))
                read(5) weights%conv1d_weight(l)%m
                end do

                if (verbose) then
                        print *, "loaded convolution weights:", n_layers, shape(weights%conv1d_weight(1)%m)
                end if

                allocate(weights%conv1d_bias(n_layers))
                do l = 1,n_layers
                allocate(weights%conv1d_bias(l)%m(d_inner))
                read(5) weights%conv1d_bias(l)%m
                end do

                if (verbose) then
                        print *, "loaded convolution bias:", n_layers, shape(weights%conv1d_bias(1)%m)
                end if

                allocate(weights%x_proj_weight(n_layers))
                do l = 1,n_layers
                allocate(weights%x_proj_weight(l)%m(d_inner,dt_rank+d_state*2))
                read(5) weights%x_proj_weight(l)%m
                end do

                if (verbose) then
                        print *, "loaded x_proj weights:", n_layers, shape(weights%x_proj_weight(1)%m)
                end if

                allocate(weights%dt_proj_weight(n_layers))
                do l = 1,n_layers
                allocate(weights%dt_proj_weight(l)%m(dt_rank,d_inner))
                read(5) weights%dt_proj_weight(l)%m
                end do

                if (verbose) then
                        print *, "loaded delta projection weights:", n_layers, shape(weights%dt_proj_weight(1)%m)
                end if

                allocate(weights%dt_proj_bias(n_layers))
                do l = 1,n_layers
                allocate(weights%dt_proj_bias(l)%m(d_inner))
                read(5) weights%dt_proj_bias(l)%m
                end do

                if (verbose) then
                        print *, "loaded delta projection bias:", n_layers, shape(weights%dt_proj_bias(1)%m)
                end if

                allocate(weights%A_log(n_layers))
                do l = 1,n_layers
                allocate(weights%A_log(l)%m(d_state,d_inner))
                read(5) weights%A_log(l)%m
                end do

                if (verbose) then
                        print *, "loaded A_log weights:", n_layers, shape(weights%A_log(1)%m)
                end if

                allocate(weights%out_proj_weight(n_layers))
                do l = 1,n_layers
                allocate(weights%out_proj_weight(l)%m(d_inner,d_model))
                read(5) weights%out_proj_weight(l)%m
                end do

                if (verbose) then
                        print *, "loaded out proj weights:", n_layers, shape(weights%out_proj_weight(1)%m)
                end if

                allocate(weights%norm_weight(n_layers))
                do l = 1,n_layers
                allocate(weights%norm_weight(l)%m(d_model))
                read(5) weights%norm_weight(l)%m
                end do

                if (verbose) then
                        print *, "loaded mixer norm weights:", n_layers, shape(weights%norm_weight(1)%m)
                end if

                allocate(weights%norm_f_weight(d_model))
                read(5) weights%norm_f_weight

                if (verbose) then
                        print *, "loaded f norm weights:", shape(weights%norm_f_weight)
                end if

                allocate(weights%wcls(emb_dim,vocab_size))
                read(5) weights%wcls

                if (verbose) then
                        print *, "loaded classifer weights:", shape(weights%wcls)
                end if
               
                read(5, iostat = error) dummy(1)

                if (verbose) then
                        if (error /= iostat_end) then
                                print *, "did not reach eof, check weights"
                        end if

                end if

        close(5) 

        end if

        if (verbose) then
                print *, "Loaded weights"
        end if



        ! state dict
        allocate(s%conv_state(n_layers))
        allocate(s%ssm_state(n_layers))
        s%seqlen_offset = 1

        ! should be able to just bulk initalize them
        do l=1,n_layers
        allocate(s%conv_state(l)%m(d_conv,d_model*expand))
        allocate(s%ssm_state(l)%m(d_state,d_model*expand))
        s%conv_state(l)%m = 0
        s%ssm_state(l)%m = 0
        end do



        if (arg_values%tokenizer /= "") then
        ! read in token vocab
        open(UNIT=5, FILE=arg_values%tokenizer, FORM="UNFORMATTED",&
               & ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")

                read(5) max_len

                ! in fortran, all strings have to be the same length
                if (allocated(vocab)) then
                        deallocate(vocab)
                        deallocate(scores)
                        deallocate(vocab_len)
                end if
                allocate(character(len=max_len) ::  vocab(vocab_size))
                allocate(scores(vocab_size))
                allocate(vocab_len(vocab_size))


                do n = 1,vocab_size-3 ! needs to be resolved, I think it's the extra tokens


                read(5) score
                read(5) tok_len
                allocate (character(tok_len) :: tmpstr)
                read(5) tmpstr

                vocab(n) = tmpstr
                scores(n) = score
                ! we track the length of each token to preserve trailing whitespace
                vocab_len(n) = tok_len

                deallocate(tmpstr)
                end do


        close(5)
        end if

        ! main part

        temperature = arg_values%temperature
        prompt = arg_values%prompt

        seq_len = arg_values%n

        t_ms_start = 0
        ! encode the prompt
        prompt_tokens = bpe_encode(prompt)

        
        if (verbose) then
        do n=1,size(prompt_tokens)
        print *, prompt_tokens(n)
        end do
        end if 

        ! indexing starts at 1, s is <s> BOS token        
            
        token = prompt_tokens(1)
        
        ! autoregressive model. get the next token from the last
        do pos = 1,seq_len
                
                logits = next_token(token,pos,s,weights)
              
                if (pos <= size(prompt_tokens)) then

                        token = prompt_tokens(pos)

                else
                        if (temperature == 0) then
                                token = maxloc(logits,DIM=1)
                        else
                                logits = softmax(logits/temperature,vocab_size)
                                token = sample(logits)
                        end if
                end if

                ! here we kept track of the length to display each token properly
                !print *, token
                write (*,fmt="(A)", advance="no") vocab(token)(1:vocab_len(token))
                
                ! start after first token as in llama2.c
                if (t_ms_start == 0) then
                        t_ms_start = time_ms()
                end if
        end do
        t_ms_end = time_ms()
        print *,""
        print *, "Inference time: ", (t_ms_end-t_ms_start)/1000, " seconds" 
        print *, 1000*(seq_len-1)/(t_ms_end-t_ms_start), "tokens/second"
        print *, "Timings"
        do l = 1,5
                print *, l, s%times(l)/seq_len
        end do
        ! end of __main__       

! functions 
contains 


        function time_ms() result(t_ms)
                real(kind=wp) :: t_ms
                integer(4) :: ms
                !call cpu_time(t_ms)
                call system_clock(ms)
                t_ms = real(ms)
        end function



        ! sample from softmax probabilities  
        function sample(p) result(i)
              real(kind=wp) :: p(:)
              integer :: i
              real(kind=wp) :: r, cdf

              call random_number(r)
              cdf = 0

              do i=1,size(p)
              cdf = cdf + p(i)
              if (r<cdf) then
                      return
              end if
              
              end do

              i = size(p)
      

        end function 
      
        elemental function silu(x) result(y)
                real(kind=wp), intent(in) :: x
                real(kind=wp) :: y

                y = x / (1.0 + exp(-x))
        end function

        ! normalize and apply weigths. Note fortran built in dot product    
        function rmsnorm(x,w) result(xr)
              real(kind=wp) :: x(:), w(:)
              real(kind=wp) :: xr(size(x))
              real(kind=wp) :: xn
              xn = sqrt(dot_product(x,x)/size(x)+1e-5)

              xr = x*w/xn
        end function

      
        pure function softmax(x,s) result (p)
              real(kind=wp), intent(in) :: x(:)
              integer, intent(in) :: s
              real(kind=wp) :: p(size(x))
              real(kind=wp) :: xi(s)

              p(:) = 0
              xi = exp(x(:s)-maxval(x(:s)))
              p(:s) = xi/sum(xi) 

        end function 

        function next_token(token, pos, s, w) result(logits)
                integer, intent(in) :: token, pos
                !type(Config), intent(in) :: p
                type(Runstate) :: s
                type(MambaWeights), intent(in) :: w
                real(kind=wp) :: logits(vocab_size)

                real(kind=wp) :: x(emb_dim), hidden_states(emb_dim)
                real(kind=wp) :: xb(emb_dim), residual(emb_dim)

                ! embeddings
                real(kind=wp), target :: qkv(emb_dim+2*kv_head_size)
                real(kind=wp), pointer :: q(:), k(:), v(:)
      
                ! position encoding  
                real(kind=wp) :: q0, q1, k0, k1, fcr, fci, v0, v1, freq, rval
                integer :: head_dim

                ! attention
                real(kind=wp) :: q_t(emb_dim/n_heads)
                real(kind=wp) :: k_t(emb_dim/n_heads)
                real(kind=wp) :: v_t(emb_dim/n_heads)
                real(kind=wp) :: xbh(emb_dim/n_heads)
                real(kind=wp) :: a
                integer :: kv_mul

                ! fc layers
                real(kind=wp), target :: hb13(2*hidden_dim)
                real(kind=wp), pointer :: hb(:), hb2(:) 

                ! counters etc
                integer :: l, i, h, t, head_size, ix
                real(kind=wp) :: time


                head_size = emb_dim/n_heads

                logits(:) = 0

                ! get embedding
                hidden_states = w%token_embedding_table(:,token)
                residual = 0
                do l = 1,n_layers
                        
                        ! embed and project
                        time = time_ms()
                        
                        ! fused add_norm

                        xb = rmsnorm(hidden_states+residual,w%norm_weight(l)%m) 
                        residual = hidden_states + residual

                        hidden_states = xb ! must be a better way
                        ! checked

                        hidden_states = mamba(hidden_states,l,s,w)

                        s%times(1) = s%times(1) + (time_ms()-time)


                end do

                hidden_states = rmsnorm(hidden_states+residual,w%norm_f_weight)

                time = time_ms()

                do ix = 1,vocab_size
                        logits(ix) = dot_product(hidden_states,w%wcls(:,ix))
                end do
                !end if
                s%times(5) = s%times(5) + (time_ms() - time)

        end function

        function mamba(hidden_states,layer,s,w) result(s_out)
                real(kind=wp), intent(in) :: hidden_states(emb_dim)
                real(kind=wp)  :: s_out(emb_dim)
                integer, intent(in) :: layer
                type(Runstate) :: s
                type(MambaWeights), intent(in) :: w
                
                real(kind=wp), allocatable :: conv_state(:,:), ssm_state(:,:)
                real(kind=wp) :: xz(2*d_inner)
                real(kind=wp), allocatable :: x(:), z(:)

                real(kind=wp) :: x_db(dt_rank+2*d_state)
                real(kind=wp), allocatable :: dt(:), A(:,:), B(:), C(:)
                real(kind=wp) :: dA(d_state,d_inner)
                real(kind=wp) :: dB(d_state,d_inner)
                real(kind=wp) :: y(d_inner)

                integer :: i,j,k


                !print *, "heres"
                conv_state = s%conv_state(layer)%m
                ssm_state = s%ssm_state(layer)%m
                 
                if (s%seqlen_offset > 0) then ! change to 1 for init
                        xz = matmul(hidden_states,w%in_proj_weight(layer)%m)
                        x = xz(1:d_inner)
                        z = xz((d_inner+1):(2*d_inner))
                        

                        ! d_conv,d_model*expand
                        conv_state(1:(d_conv-1),:) = conv_state(2:,:)
                        conv_state(d_conv,:) = x 
                        x = sum (conv_state * reshape(w%conv1d_weight(layer)%m,[d_conv,d_inner]),1)
                        
                        x = x + w%conv1d_bias(layer)%m 
                        x = silu(x)
                        x_db = matmul(x,w%x_proj_weight(layer)%m)
                        
                        dt = x_db(1:dt_rank)
                        B = x_db((dt_rank+1):(dt_rank+d_state))
                        C = x_db((dt_rank+d_state+1):(dt_rank+2*d_state))

                        dt = matmul(dt,w%dt_proj_weight(layer)%m)
                        A = -exp(w%A_log(layer)%m)

                        dt = log(1+exp(dt + w%dt_proj_bias(layer)%m)) !softplus
                        
                        do j=1,d_inner !size of dt
                        do k=1,d_state
                        dA(k,j) = dt(j) * A(k,j)
                        end do
                        end do


                        dA = exp(dA)

                        do j=1,d_inner !dt
                        do k=1,d_state !B
                        dB(k,j) = dt(j)*B(k)
                        end do
                        end do

                        ssm_state = ssm_state * dA + spread(x,1,d_state) * dB
                       
                        y=0
                        do j=1,d_state
                        do k=1,d_inner
                        y(k) = y(k) + ssm_state(j,k)*C(j)
                        end do
                        end do 
                        y = y + w%D(layer)%m * x
                        y = y * silu(z)
                        s%conv_state(layer)%m = conv_state
                        s%ssm_state(layer)%m = ssm_state
                        
                        s_out = matmul(y,w%out_proj_weight(layer)%m)

                end if

        end function

        ! lookup the encoding of a token
        function lookup(s,l) result(ind)
                character(len=*) :: s
                integer :: l
                integer :: i, ind

                do i = 1,size(vocab)
                if (vocab(i) == s .and. vocab_len(i)==l) then
                        ind = i
                        return
                end if
                end do
                ind = -1
        end function

        ! encode text into tokens
        function bpe_encode(text) result(tokens)

                character(len=*) :: text
                integer, allocatable :: tokens(:)
                integer, allocatable :: tmp_tokens(:)
                integer :: i, ind, best_id, t1, t2
                real(kind=wp) :: score, best_score
                character(:), dimension(:), allocatable :: running_merge
                integer, allocatable :: running_merge_len(:)

                allocate(tokens(len(text)))

                do i = 1,len(text)
                        tokens(i) = lookup(text(i:i), 1)
                end do


                do while(1==1)

                        allocate(character(len=2*max_len) :: running_merge(size(tokens)-1))
                        allocate(running_merge_len(size(tokens)-1))

                        do i=1,(size(tokens)-1)

                                ! don't use trim, slice of to vocab_len
                                ! need to keep track of the true length of everything
                                t1 = vocab_len(tokens(i))
                                t2 = vocab_len(tokens(i+1))
                                running_merge(i) = vocab(tokens(i))(1:t1)//vocab(tokens(i+1))(1:t2)
                                running_merge_len(i) = t1+t2
                        end do


                        best_id = -1
                        best_score = -1e10
                        do i = 1,(size(tokens)-1)
                                ind = lookup(running_merge(i), running_merge_len(i))
                                if (ind > 0) then
                                        score = scores(ind)
                                        if (score > best_score) then
                                                best_score = score
                                                best_id = i
                                        end if

                                end if
                        end do


                        if (best_id == -1) then
                                exit
                        end if
                        
                        allocate(tmp_tokens(size(tokens)-1))
                        tmp_tokens(1:(best_id-1)) = tokens(1:(best_id-1))
                        tmp_tokens(best_id) = lookup(running_merge(best_id),running_merge_len(best_id))
                        tmp_tokens((best_id+1):) = tokens((best_id+2):)
                        deallocate(tokens)
                        call move_alloc(tmp_tokens,tokens)


                        deallocate(running_merge)
                        deallocate(running_merge_len)

                end do


        end function



end program ssm_inference
