module pretokenize

        character(:), allocatable :: result, orig
        character(:), dimension(:), allocatable :: c_encoding
        integer :: j

        !c_encoding = make_encoding()
        

        !result = pre_tokenize('Andy99아마')

        !print *, result

        !orig = decode(result)





contains 

        subroutine init
                c_encoding = make_encoding()
        end subroutine
        
        function make_encoding() result(output)
        character(len=2) ::  output(256)
        integer(1) :: t
        integer :: i, k

        character(len=:), allocatable :: ext 


        ext = 'ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠ!"#$%&' // "'"&
        & // '()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUV'&
                & //'WXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'&
                & // 'ġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł¡¢£¤¥'&
                & // '¦§¨©ª«¬Ń®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌ'&
                & // 'ÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ'

        !print *, len(ext)
        
        ! move one character at a time into output
        i = 1
        k = 1
        do while (i <= len(ext)) 
        read(ext(i:i), "(A)") t

        if (t >= 0 .and. t < 128) then
        output(k) = ext(i:i)
        i = i + 1
        else
        output(k) = ext(i:i+1)
        i = i + 2        
        end if
        k = k + 1

        end do
        

        end function
        

        function decode(r,l) result (tmp_str)
        character(len=*) :: r
        integer :: l
        !character(len=len(r)) :: s
        integer :: i, j, k
        integer(1) :: t
        character(len=2) :: next 
        integer(1), allocatable :: bytes(:)
        character(len=:), allocatable :: tmp_str

        !c_encoding = make_encoding()

        i=1
        j=1
        allocate(bytes(l))

        do while (i<=l)
        read(r(i:i), "(A)") t
        if (t >= 0 .and. t < 128) then
        next = r(i:i)
        i = i + 1
        else
        next = r(i:i+1)
        i = i + 2
        end if
        !j = j + 1
        !print *, next

        ! we need the number corresponding to next's location in the encoding
        !k = 1
                
        do k = 1,size(c_encoding)
        if (c_encoding(k) == next) then
                ind = k
                exit
        end if
        end do

        !print *, ind
        if (ind < 128) then
        bytes(j) = ind - 1
        else
                bytes(j) = ind-256-1
        end if
        j = j + 1
        end do
        !print *, bytes
        ! open and write to a temp file
        open(UNIT=5, FILE="tempfile.xxx", FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="REPLACE", POSITION="REWIND", ACTION="WRITE")
        write(5) bytes
        close(5)

        allocate(character(len=j-1) :: tmp_str)

        open(UNIT=5, FILE="tempfile.xxx", FORM="UNFORMATTED",&
                &ACCESS="STREAM", STATUS="OLD", POSITION="REWIND", ACTION="READ")
        read(5) tmp_str
        close(5)
        
        
        !print *, tmp_str
        
        end function

        function pre_tokenize(s) result (r)
        ! s is a unicode string
        ! we can get the bytes just by indexing
        character(len=*) :: s
        character(len=len(s)*2) :: r
        integer :: i
        integer(1) :: t

        r = ""

        !print *, len(s)

        do i = 1,len(s)
                read(s(i:i), "(A)") t
                ! look it up
                if (t < 0) then
                        !print *, c_encoding(t+256+1)
                        r = trim(r) // trim(c_encoding(t+256+1)) 
                else
                        r = trim(r) // trim(c_encoding(t+1))        
                end if
        
        
                !print *, t
                !print *, c_encoding(t+256)
        end do

        r = trim(r)
        !print *, r

        end function


end module
