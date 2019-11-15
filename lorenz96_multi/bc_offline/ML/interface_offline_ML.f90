module interface_ML
  use common
  use lorenz96
  use forpy_mod

  implicit none

  type(module_py) :: fpi
  type(object) :: r_vec, model
  type(ndarray) :: vec, r_vec_cast
  type(list) :: paths
  type(tuple) :: args

contains
!------------------------------------------------!
subroutine ML_initialize
  integer :: ierror
  ierror = forpy_initialize()
  ierror = get_sys_path(paths)
  ierror = paths%append(".")
  ierror = import_py(fpi, "forpy_interface")
 !Get the restored model
!  ierror = call_py(model, fpi, "get_model")

return
end subroutine ML_initialize
!------------------------------------------------!
subroutine ML_predict(f_inp,f_vec)
  integer :: ierror
  real(r_size),intent(in) :: f_inp(nx)
  real(r_size),intent(out) :: f_vec(nx)

  !This part would go in the forecast-analysis loop
    !Assuming one dimensional input fortran array containg all the variable

!    ierror = ndarray_create(vec, f_inp)
   !Sending the array for new_forecast
!    ierror = tuple_create(args, 2)
!    ierror = args%setitem(0, model)
!    ierror = args%setitem(1, vec)
!    ierror = call_py(r_vec, fpi, "prediction", args)
!    ierror = cast(r_vec_cast, r_vec)
    !Transferring the data to fortran vector for data assimilation
!    ierror = r_vec_cast%get_data(f_vec)
!    ierror = vec%get_data(f_vec)


f_vec=f_inp

return
end subroutine ML_predict    
!------------------------------------------------!

subroutine ML_finalize
  
!  call args%destroy
!  call r_vec%destroy
!  call model%destroy
  call paths%destroy
!  call vec%destroy
!  call r_vec_cast%destroy

  call forpy_finalize

return
end subroutine ML_finalize
!------------------------------------------------!

end module interface_ML
