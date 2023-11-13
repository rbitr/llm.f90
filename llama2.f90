! llama2.f90


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
                        arg_values%model_file = "stories15M.bin"
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



program llama2 
        use iso_c_binding
        use precision_module
        use weight_module
        use arg_parse
        use read_ggml, only: load_ggml
        !use omp_lib

        implicit none


        
        ! weights and states
        integer :: dummy(7)
        !integer :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        integer, parameter :: emb_dim = 2048
        integer, parameter :: hidden_dim = 5632
        integer, parameter :: n_layers = 22
        integer, parameter :: n_heads = 32
        integer, parameter :: n_kv_heads = 4
        integer, parameter :: vocab_size = 32000
        integer :: seq_len = 2048

        type(TransformerWeights) :: weights
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


        call parse_args(arg_values)

        verbose = arg_values%verbose
        
        if (.not. arg_values%ak) then

        call load_ggml(arg_values%model_file, weights, dummy_conf, vocab, scores, vocab_len, verbose)
        max_len = maxval(vocab_len)-2
        head_size = emb_dim / n_heads
        kv_head_size = n_kv_heads * head_size
        shared_weights =.false.


        ! open the model file 
        else
        open(UNIT=5, FILE=arg_values%model_file, FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
                ! config
                !read(5) emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
                read(5) dummy

                head_size = emb_dim / n_heads
                kv_head_size = n_kv_heads * head_size
                
                if (verbose) then
                        print *, "Embedding dimension: ", emb_dim
                        print *, "Hidden dimension: ", hidden_dim
                        print *, "Layers: ", n_layers
                        print *, "Heads: ", n_heads
                        print *, "kv Heads: ", n_kv_heads
                        print *, "Vocabulary Size: ", vocab_size
                        print *, "Sequence Length: ", seq_len
                        print *, "Head Size: ", head_size
                        print *, "kv Head Size: ", kv_head_size

                end if 

                shared_weights = .false.
                !if (vocab_size > 0) then 
                !        shared_weights = .true.
                !else
                !        shared_weights = .false.
                !        vocab_size = -vocab_size
                !end if 

                
                allocate(weights%token_embedding_table(emb_dim,vocab_size))
                read(5) weights%token_embedding_table
                
                if (verbose) then
                        print *, "loaded embedding weights:", size(weights%token_embedding_table)
                end if 

                allocate(weights%rms_att_weight(emb_dim,n_layers))
                read(5) weights%rms_att_weight

                if (verbose) then
                        print *, "loaded rms att weights:", size(weights%rms_att_weight)
                end if

                !!!!!!!!!
                ! single qkv
                !!!!!!!!!
                allocate(weights%wqkv(emb_dim,emb_dim+2*kv_head_size,n_layers))
                do l = 1,n_layers
                read(5) weights%wqkv(:,1:emb_dim,l)
                end do

                if (verbose) then
                        print *, "loaded wq weights:", size(weights%wqkv(:,1:emb_dim,:))
                end if

                do l = 1,n_layers
                read(5) weights%wqkv(:,(emb_dim+1):(emb_dim+kv_head_size),l)
                end do

                if (verbose) then
                        print *, "loaded wk weights:", size(weights%wqkv(:,(emb_dim+1):(emb_dim+kv_head_size),:))
                end if

                do l = 1,n_layers
                read(5) weights%wqkv(:,(emb_dim+kv_head_size+1):,l)
                end do

                if (verbose) then
                        print *, "loaded wv weights:", size(weights%wqkv(:,(emb_dim+kv_head_size+1):,l))
                end if
                
                allocate(weights%wo(emb_dim,emb_dim,n_layers))
                do l = 1,n_layers
                read(5) weights%wo(:,:,l)
                end do

                if (verbose) then
                        print *, "loaded wo weights:", size(weights%wo)
                end if

                allocate(weights%rms_ffn_weight(emb_dim,n_layers))
                read(5) weights%rms_ffn_weight

                if (verbose) then
                        print *, "loaded rms ffn  weights:", size(weights%rms_ffn_weight)
                end if

                allocate(weights%w13(emb_dim,2*hidden_dim,n_layers))
                do l = 1,n_layers
                read(5) weights%w13(:,1:hidden_dim,l)
                end do

                if (verbose) then
                        print *, "loaded w1 weights:", size(weights%w13(:,1:hidden_dim,:))
                end if

                allocate(weights%w2(hidden_dim,emb_dim,n_layers))
                do l = 1,n_layers
                read(5) weights%w2(:,:,l)
                end do

                if (verbose) then
                        print *, "loaded w2 weights:", size(weights%w2)
                end if

                do l = 1,n_layers
                read(5) weights%w13(:,(hidden_dim+1):,l)
                end do

                if (verbose) then
                        print *, "loaded w3 weights:", size(weights%w13(:,hidden_dim,:))
                end if

                allocate(weights%rms_final_weight(emb_dim))
                read(5) weights%rms_final_weight

                if (verbose) then
                        print *, "loaded rms_final weights:", size(weights%rms_final_weight)
                end if

                if (.not. shared_weights) then
                        allocate(weights%wcls(emb_dim,vocab_size))
                        read(5) weights%wcls

                        if (verbose) then
                                print *, "loaded wcls weights:", size(weights%wcls)
                        end if

                end if

        close(5) 

        end if

        if (verbose) then
                print *, "Loaded weights"
        end if

        ! config
        conf%emb_dim = emb_dim
        conf%hidden_dim = hidden_dim
        conf%n_layers = n_layers
        conf%n_heads = n_heads
        conf%n_kv_heads = n_kv_heads
        conf%vocab_size = vocab_size
        conf%seq_len = seq_len
        conf%kv_head_size = kv_head_size

        ! state dict
        allocate(s%att(seq_len,n_heads))
        allocate(s%key_cache(kv_head_size,seq_len,n_layers))
        allocate(s%value_cache(kv_head_size,seq_len,n_layers))

        ! not needed
        s%att(:,:) = 0
        s%key_cache(:,:,:) = 0
        s%value_cache(:,:,:) = 0
        s%times = 0

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


                do n = 1,vocab_size


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

        if (arg_values%n <= seq_len) then
                seq_len = arg_values%n
        else
                print *, arg_values%n, "greater than maxinum squence length"
                print *, "set to", seq_len
        end if

        t_ms_start = 0
        ! encode the prompt
        prompt_tokens = bpe_encode(prompt)

        ! indexing starts at 1, s is <s> BOS token        
            
        token = 2
        
        ! autoregressive model. get the next token from the last
        do pos = 1,seq_len
                logits = transformer(token,pos,s,weights)
              
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
      
        ! normalize and apply weigths. Note fortran built in dot product    
        function rmsnorm(x,w) result(xr)
              real(kind=wp) :: x(:), w(:)
              real(kind=wp) :: xr(size(x))
              real(kind=wp) :: xn
              xn = sqrt(dot_product(x,x)/size(x)+1e-5)

              xr = x*w/xn
        end function

        !function trmsnorm(x,w,xn) result(xr)
        !      real(kind=wp) :: x, w
        !      real(kind=wp) :: xr
        !      real(kind=wp) :: xn
        !      !xn = sqrt(dot_product(x,x)/size(x)+1e-5)
        !
        !      xr = x*w/xn
        !end function
      
        pure function softmax(x,s) result (p)
              real(kind=wp), intent(in) :: x(:)
              integer, intent(in) :: s
              real(kind=wp) :: p(size(x))
              real(kind=wp) :: xi(s)

              p(:) = 0
              xi = exp(x(:s)-maxval(x(:s)))
              p(:s) = xi/sum(xi) 

        end function 

        function transformer(token, pos, s, w) result(logits)
                integer, intent(in) :: token, pos
                !type(Config), intent(in) :: p
                type(Runstate) :: s
                type(TransformerWeights), intent(in) :: w
                real(kind=wp) :: logits(vocab_size)

                real(kind=wp) :: x(emb_dim)
                real(kind=wp) :: xb(emb_dim)

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

                ! convert precision        
                x = w%token_embedding_table(:,token)


                do l = 1,n_layers
                        
                        ! embed and project
                        time = time_ms()
                        xb = rmsnorm(x,w%rms_att_weight(:,l))
                        
                        do ix = 1,size(qkv)
                        qkv(ix) = dot_product(xb,(w%wqkv(:,ix,l)))
                        end do
                        
                        q => qkv(1:emb_dim)
                        k => qkv((emb_dim+1):(emb_dim+kv_head_size))
                        v => qkv((emb_dim+kv_head_size+1):(emb_dim+2*kv_head_size))
                        
                        
                        s%times(1) = s%times(1) + (time_ms()-time)
                        ! position encoding
        
                        time = time_ms()
                        ! check that this doens't add any time
                        do i=1,emb_dim,2
                                head_dim = mod(i,head_size)
                                freq = 1.0 / (10000.0 ** (real(head_dim,kind=wp) / head_size))
                                rval = pos * freq
                                fcr = cos(rval)
                                fci = sin(rval)
                                q0 = q(i)
                                q1 = q(i+1)
                                q(i) = q0 * fcr - q1 * fci
                                q(i+1) = q0 * fci + q1 * fcr
                                if (i<kv_head_size) then
                                        k0 = k(i)
                                        k1 = k(i+1)
                                        k(i) = k0 * fcr - k1 * fci
                                        k(i+1) = k0 * fci + k1 * fcr
                                end if
                        end do
                        
                        s%times(2) = s%times(2) + (time_ms()-time)

                        ! cache k and v for this position
                        s%key_cache(:,pos,l) = k
                        s%value_cache(:,pos,l) = v 

                        xb(:) = 0
                        
                        ! multi head attention and fc layers
                        time = time_ms()
                        
                        kv_mul = n_heads / n_kv_heads
                        
                        do h = 0,(n_heads-1)        

                        q_t = q((h*head_size+1):((h+1)*head_size))
          
                        do t = 1,(pos)
                        !k_t = s%key_cache((h*head_size+1):((h+1)*head_size),t,l)
                        ! for shared heads 
                        k_t = s%key_cache(((h/kv_mul)*head_size+1):(((h+1)/kv_mul)*head_size),t,l)
                        s%att(t,h+1) = dot_product(q_t,k_t)/sqrt(real(head_size,wp))
                        end do  
          
                        ! beginning to POS, inclusive. so if pos = 1, there is 1...      
                        s%att(:,h+1) = softmax(s%att(:,h+1),pos)
                        xbh(:) = 0  
                        
                        do t = 1,(pos) 
                        !v_t = s%value_cache((h*head_size+1):((h+1)*head_size),t,l)      
                        v_t = s%value_cache(((h/kv_mul)*head_size+1):(((h+1)/kv_mul)*head_size),t,l) 
                        a = s%att(t,h+1)
                        xbh = xbh + a*v_t  
                        end do     
        
                        xb((h*head_size+1):((h+1)*head_size)) = xbh
                       
                        end do  
                        s%times(3) = s%times(3) + (time_ms() - time)

                        time = time_ms()
                        
                        do ix=1,emb_dim
                        x(ix) = x(ix) + dot_product(xb,w%wo(:,ix,l))
                        end do
                        

                        xb = rmsnorm(x,w%rms_ffn_weight(:,l))
          
                        do ix = 1,size(hb13)
                        hb13(ix) = dot_product(xb,w%w13(:,ix,l))
                        end do
                        hb => hb13(1:hidden_dim)
                        hb2 => hb13((hidden_dim+1):(2*hidden_dim))
                        hb = hb*(1/(1+exp(-hb)))
                        hb = hb*hb2

                        do ix = 1,emb_dim
                        x(ix) = x(ix) + dot_product(hb,w%w2(:,ix,l))
                        end do

                        s%times(4) = s%times(4) + (time_ms() - time)

                end do

                time = time_ms()
                x = rmsnorm(x, w%rms_final_weight)

      
      
                !if (shared_weights) then
                !        logits = vm_matmul(x,(w%token_embedding_table))
                !else
                        do ix = 1,vocab_size
                                logits(ix) = dot_product(x,w%wcls(:,ix))
                        end do
                !end if
                s%times(5) = s%times(5) + (time_ms() - time)

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



end program llama2
