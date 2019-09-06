function instance = get_element_instance(str)
    switch lower(str)
       case 'tanh'
          instance = TanhMap();
       case 'add'
          instance = AddGate();
        case 'multiply'
          instance = MulGate();
        case 'purelin'
          instance = PurelinMap();
       case 'sigmoid'
          instance = SigmoidMap();
        case 'quadratic'
          instance = QuadraticLoss();                  
        case 'relu'
          instance =  ReLUMap();
       otherwise
          error('cannot recogize %s in get_gate_instance function',str);
    end
end