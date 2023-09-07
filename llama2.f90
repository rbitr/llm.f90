! llama2.f90
! implementation of a toy llama model

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
                real(kind=wp), allocatable :: wq(:,:,:)
                real(kind=wp), allocatable :: wk(:,:,:)
                real(kind=wp), allocatable :: wv(:,:,:)
                real(kind=wp), allocatable :: wo(:,:,:)
                real(kind=wp), allocatable :: w1(:,:,:)
                real(kind=wp), allocatable :: w2(:,:,:)
                real(kind=wp), allocatable :: w3(:,:,:)
                real(kind=wp), allocatable :: rms_final_weight(:)
                real(kind=wp), allocatable :: freq_cis_real(:,:)
                real(kind=wp), allocatable :: freq_cis_imag(:,:)
  
        end type TransformerWeights

        type Config
                INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        end type Config

        type RunState

                real(kind=wp), allocatable :: att(:,:)
                real(kind=wp), allocatable :: key_cache(:,:,:)
                real(kind=wp), allocatable :: value_cache(:,:,:)

        end type RunState

end module weight_module

module arg_parse
        implicit none

        type args
                real :: temperature
                character(:), allocatable :: model_file
                character(:), allocatable :: prompt
                logical :: verbose
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
                                                case default
                                                print *, 'Unrecognized option:', trim(arg)
                                                end select
                        end do

                        ! check for arguments


                end subroutine

end module arg_parse

program llama2 
        use precision_module
        use weight_module
        use arg_parse
        
        ! weights and states
        INTEGER :: emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        type(TransformerWeights) :: weights
        integer :: head_size, tmp
        type(config) :: conf
        type(RunState) :: s
        real(kind=wp), allocatable :: logits(:)
        real(kind=wp), allocatable :: freq_buf(:)

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

        ! open the model file 
        open(UNIT=5, FILE=arg_values%model_file, FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
                ! config
                read(5) emb_dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
                
                if (verbose) then
                        print *, "Embedding dimension: ", emb_dim
                        print *, "Hidden dimension: ", hidden_dim
                        print *, "Layers: ", n_layers
                        print *, "Heads: ", n_heads
                        print *, "kv Heads: ", n_kv_heads
                        print *, "Vocabulary Size: ", vocab_size
                        print *, "Sequence Length: ", seq_len

                end if 

                ! once we know the config sizes, allocate the arrays
                allocate(weights%token_embedding_table(emb_dim,vocab_size))
                allocate(weights%rms_att_weight(emb_dim,n_layers))

                allocate(weights%wq(emb_dim,emb_dim,n_layers))
                allocate(weights%wk(emb_dim,emb_dim,n_layers))
                allocate(weights%wv(emb_dim,emb_dim,n_layers))
                allocate(weights%wo(emb_dim,emb_dim,n_layers))

                allocate(weights%rms_ffn_weight(emb_dim,n_layers))

                allocate(weights%w1(emb_dim,hidden_dim,n_layers))
                allocate(weights%w2(hidden_dim,emb_dim,n_layers))
                allocate(weights%w3(emb_dim,hidden_dim,n_layers))

                allocate(weights%rms_final_weight(emb_dim))

                head_size = emb_dim / n_heads

                allocate(weights%freq_cis_real(head_size/2,seq_len))
                allocate(weights%freq_cis_imag(head_size/2,seq_len))

                ! read everything in
                read(5) weights%token_embedding_table
                read(5) weights%rms_att_weight
                read(5) weights%wq
                read(5) weights%wk
                read(5) weights%wv
                read(5) weights%wo
                read(5) weights%rms_ffn_weight
                read(5) weights%w1
                read(5) weights%w2
                read(5) weights%w3
                read(5) weights%rms_final_weight
                read(5) weights%freq_cis_real
                read(5) weights%freq_cis_imag

        close(5) 

        ! config
        conf%emb_dim = emb_dim
        conf%hidden_dim = hidden_dim
        conf%n_layers = n_layers
        conf%n_heads = n_heads
        conf%n_kv_heads = n_kv_heads
        conf%vocab_size = vocab_size
        conf%seq_len = seq_len

        ! state dict
        allocate(s%att(seq_len,n_heads))
        allocate(s%key_cache(emb_dim,seq_len,n_layers))
        allocate(s%value_cache(emb_dim,seq_len,n_layers))

        ! not needed
        s%att(:,:) = 0
        s%key_cache(:,:,:) = 0
        s%value_cache(:,:,:) = 0

        ! read in token vocab
        open(UNIT=5, FILE="tokenizer.bin", FORM="UNFORMATTED", ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")

                read(5) max_len

                ! in fortran, all strings have to be the same length
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

        ! __main__ part
        ! argparse
        num_args = command_argument_count()


        temperature = arg_values%temperature
        prompt = arg_values%prompt

        if (arg_values%n < seq_len) then
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
                logits = transformer(token,pos,conf,s,weights)
              
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
        print *, 1000*seq_len/(t_ms_end-t_ms_start), "tokens/second"
        ! end of __main__       

! functions 
contains 

        function time_ms() result(t_ms)
                real(kind=wp) :: t_ms
                integer :: times(8)

                call date_and_time(values=times)

                t_ms = (times(5)*3600. + times(6)*60. + times(7)) * 1000. + times(8)
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
      
        ! declared as "pure" for potentiall parallelization of heads
        pure function softmax(x,s) result (p)
              real(kind=wp), intent(in) :: x(:)
              integer, intent(in) :: s
              real(kind=wp) :: p(size(x))
              real(kind=wp) :: xi(s)

              p(:) = 0
              xi = exp(x(:s)-maxval(x(:s)))
              p(:s) = xi/sum(xi) 

        end function 

      ! where the action isa
      ! see https://jaykmody.com/blog/gpt-from-scratch/ and ref therin for how attention works
        function transformer(token, pos, p, s, w) result(logits)
                integer, intent(in) :: token, pos
                type(Config), intent(in) :: p
                type(Runstate) :: s
                type(TransformerWeights), intent(in) :: w
                real(kind=wp) :: logits(p%vocab_size)

                real(kind=wp) :: x(p%emb_dim)
                real(kind=wp) :: xb(p%emb_dim)
                real(kind=wp) :: freq_cis_real_row(p%emb_dim/p%n_heads/2)
                real(kind=wp) :: freq_cis_imag_row(p%emb_dim/p%n_heads/2)

                ! embeddings
                real(kind=wp) :: q(emb_dim)
                real(kind=wp) :: k(emb_dim)
                real(kind=wp) :: v(emb_dim)
      
                ! position encoding  
                real(kind=wp) :: q0, q1, k0, k1, fcr, fci

                ! attention
                real(kind=wp) :: q_t(p%emb_dim/p%n_heads)
                real(kind=wp) :: k_t(p%emb_dim/p%n_heads)
                real(kind=wp) :: v_t(p%emb_dim/p%n_heads)
                real(kind=wp) :: xbh(p%emb_dim/p%n_heads)
                real(kind=wp) :: a

                ! fc layers
                real(kind=wp) :: hb(hidden_dim)
                real(kind=wp) :: hb2(hidden_dim)
      
                ! counters etc
                integer :: l, i, h, t, head_size


                head_size = p%emb_dim/p%n_heads

                logits(:) = 0

                x = w%token_embedding_table(:,token)

                freq_cis_real_row = w%freq_cis_real(:,pos)
                freq_cis_imag_row = w%freq_cis_imag(:,pos)

                do l = 1,p%n_layers
                        
                        ! embed and project
                        xb = rmsnorm(x,w%rms_att_weight(:,l)) 
        
                        q = matmul(xb,w%wq(:,:,l))
                        k = matmul(xb,w%wk(:,:,l))
                        v = matmul(xb,w%wv(:,:,l))
       
                        ! position encoding
        
                        do h = 0,(p%n_heads-1)
                                do i = 1,head_size,2

                                q0 = q(h*head_size+i)  
                                q1 = q(h*head_size+i+1)
                                k0 = k(h*head_size+i)
                                k1 = k(h*head_size+i+1)
                                fcr = freq_cis_real_row((i-1)/2+1)
                                fci = freq_cis_imag_row((i-1)/2+1)
                                q(h*head_size+i) = q0 * fcr - q1 * fci
                                q(h*head_size+i+1) = q0 * fci + q1 * fcr
                                k(h*head_size+i) = k0 * fcr - k1 * fci
                                k(h*head_size+i+1) = k0 * fci + k1 * fcr
          
                                end do
                        end do

                        ! cache k and v for this position
                        s%key_cache(:,pos,l) = k
                        s%value_cache(:,pos,l) = v 

                        xb(:) = 0
                        
                        ! multi head attention and fc layers
                        do h = 0,(p%n_heads-1)        
                        ! alternately uncomment below to make explicitly concurrent 
                        !do concurrent (h =1:(p%n_heads-1))

                        q_t = q((h*head_size+1):((h+1)*head_size))
          
                        do t = 1,(pos)
                        k_t = s%key_cache((h*head_size+1):((h+1)*head_size),t,l)
                        s%att(t,h+1) = dot_product(q_t,k_t)/sqrt(real(head_size,wp))
                        end do  
          
                        ! beginning to POS, inclusive. so if pos = 1, there is 1...      
                        s%att(:,h+1) = softmax(s%att(:,h+1),pos)
                        xbh(:) = 0  
                        
                        do t = 1,(pos) 
                        v_t = s%value_cache((h*head_size+1):((h+1)*head_size),t,l)      
                        a = s%att(t,h+1)
                        xbh = xbh + a*v_t  
                        end do     
        
                        xb((h*head_size+1):((h+1)*head_size)) = xbh
                       
                        end do  


                        x = x + matmul(xb, w%wo(:,:,l))

                        xb = rmsnorm(x,w%rms_ffn_weight(:,l))
          

                        hb = matmul(xb,w%w1(:,:,l))
                        hb2 = matmul(xb,w%w3(:,:,l))

                        hb = hb*(1/(1+exp(-hb)))

                        hb = hb*hb2
                        xb = matmul(hb,w%w2(:,:,l))

                        x = x + xb

                end do

                x = rmsnorm(x, w%rms_final_weight)

      
      

                logits = matmul(x,w%token_embedding_table)


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
