#ifndef MY_COEFFICIENT
#define MY_COEFFICIENT

#include <coefficient.hpp>


namespace ngfem
{
  
  /*
    CoefficientFunction which computes the eigenvalues of
    a symmetric 2x2 matrix-valued CoefficientFunction
  */
  
  class EigH_CF : public T_CoefficientFunction<EigH_CF>
  {
    shared_ptr<CoefficientFunction> mat;
  public:

    // Input mat = [[a, b],
    //              [b, c]],
    // symmetric real valued matrix, where a, b, c are ngsolve CF.
    // This EigH_CF then evaluates  a 2-vector containing the two real
    // eigenvalues of mat.      
    
    EigH_CF(shared_ptr<CoefficientFunction> _mat)
      : mat(_mat)
    {

      if (_mat->Dimensions() != Array<int> ({ 2, 2}))
        throw Exception("input must be a 2x2 matrix CF");

      // we return a 2-vector, so the shape is (2,)
      SetDimensions(Array<int>({2}));
    }

    virtual Array<shared_ptr<CoefficientFunction>>
    InputCoefficientFunctions() const override    {
      return Array<shared_ptr<CoefficientFunction>>({mat});
    }

    // Evaluates this CF for all points of input integration rule ir.
    // The function is generated for the generic types T,
    // and will be instantiated for double, complex, AutoDiff, SIMD<double>, ...
    
    template <typename MIR, typename T, ORDERING ORD>
    void T_Evaluate (const MIR & ir,
                     BareSliceMatrix<T,ORD> result) const
    {
      // create a temporary matrix on the stack
      STACK_ARRAY(T, hmem, 4*ir.Size());
      FlatMatrix<T,ORD> temp(4, ir.Size(), &hmem[0]);

      // evalute input coefficient function, a 2x2 matrix
      mat->Evaluate (ir, temp);

      for (size_t i = 0; i < ir.Size(); i++)
        {
          // matrix entries are [ [a,b], [b,c] ]
          T a = temp(0,i);
          T b = temp(1,i);
          T c = temp(3,i);

	  // These are the two eigenvalues of the matrix:
          result(0,i) = 0.5*(a+c) + sqrt ( 0.25*(a-c)*(a-c) + b*b );
          result(1,i) = 0.5*(a+c) - sqrt ( 0.25*(a-c)*(a-c) + b*b );
        }
    }

    // Evaluate for all points of an integration rule.
    // The matrix is provided by caller.
    
    template <typename MIR, typename T, ORDERING ORD>
    void T_Evaluate (const MIR & ir,
                     FlatArray<BareSliceMatrix<T,ORD>> input,                       
                     BareSliceMatrix<T,ORD> result) const
    {
      auto temp = input[0];
      for (size_t i = 0; i < ir.Size(); i++)
        {
          // matrix entries are [ [a,b], [b,c] ]          
          T a = temp(0,i);
          T b = temp(1,i);
          T c = temp(3,i);
          result(0,i) = 0.5*(a+c) + sqrt ( 0.25*(a-c)*(a-c) + b*b );
          result(1,i) = 0.5*(a+c) - sqrt ( 0.25*(a-c)*(a-c) + b*b );
        }
    }


    // Compute the directional derivative of eigenvalues with respect
    // to a node "var" in the tree of expressions definining the
    // matrix coefficient function. Both "var" and "dir" should be of
    // the same dimensions.
    
    shared_ptr<CoefficientFunction> 
    Diff(const CoefficientFunction * var,
	 const shared_ptr<CoefficientFunction> dir) const override {

      // Compute the derivative of the eigenvalues by
      // 
      //       d lam = cof(lam - A) : d A  / trace(cof(lam - A))
      //
      // where A = [[a, b], [b, c]].  This formula is derived by implicit
      // differentiation of the eigenvalue equation det(lam - A) = 0.

      auto thisptr = const_pointer_cast<CoefficientFunction>
	(this->shared_from_this());
      Array<shared_ptr<CoefficientFunction>> dlam(2);

      for (int i=0; i < 2; i++) {  // for each eigenvalue, compute its derivative
	auto lam = MakeComponentCoefficientFunction(thisptr, i);
	auto cof = CofactorCF(mat - lam * IdentityCF(2));
	auto dA = mat->Diff(var, dir);
	auto tr = TraceCF(cof);
	dlam[i] = InnerProduct(cof, dA) / tr;	  
      }      
      return  MakeVectorialCoefficientFunction(std::move(dlam));	    
    }
    
  };
}



#endif

