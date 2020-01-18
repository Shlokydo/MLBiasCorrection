program test_forpy
  
  use forpy_mod
  implicit none

  integer :: ierror, j
  type(module_py) :: fpi
  type(list) :: paths
  type(object) :: plist, model, m_out_obj
  type(tuple) :: args0, args1
  type(ndarray) :: m_inp_vec, m_out_vec

  real, dimension(40) :: f_inp
  real, dimension(40), asynchronous :: f_out
  real, dimension(:), pointer :: f_poi

  ierror = forpy_initialize()

  ierror = get_sys_path(paths)
  ierror = paths%append(".")

  ierror = import_py(fpi, "forpy_interface")

  !Get the parameter list
  ierror = call_py(plist, fpi, "get_pickle")

  !Get the restored model
  ierror = tuple_create(args0, 1)
  ierror = args0%setitem(0, plist)
  ierror = call_py(model, fpi, "get_model", args0)

  ierror = tuple_create(args1, 3)
  ierror = args1%setitem(0, plist)
  ierror = args1%setitem(1, model)

  !This part would go in the forecast-analysis loop
    !Assuming one dimensional input fortran array containing all the variables
    call RANDOM_NUMBER(f_inp)
    ierror = ndarray_create(m_inp_vec, f_inp)
    ierror = args1%setitem(2, m_inp_vec)
    ierror = call_py(m_out_obj, fpi, "prediction", args1)
    ierror = cast(m_out_vec, m_out_obj)

    !Transferring the data to fortran vector for data assimilation
    ierror = m_out_vec%get_data(f_poi)
    print*, f_poi(4)
  
  call paths%destroy
  call fpi%destroy
  call plist%destroy
  call model%destroy
  call args0%destroy
  call args1%destroy
  call m_inp_vec%destroy
  call m_out_obj%destroy
  call m_out_vec%destroy
  
  call forpy_finalize

end program test_forpy
