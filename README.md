gen_m3s.cpp:
This code generates a random instance of Max3Sat in .cnf form given the number of variables N and number of clauses K.
The file is then stored in ./gen_max3sat/vNcK.cnf.

prepare_m3s.cpp:
This code takes in an instance of Max3Sat in .cnf form and converts it to a form appropriate for running htaac-qsos.
The resulting parameter files are sparse and stored in ./problem/ .

htaac_qsos.cpp:
This is the first version of the HTAAC-QSOS code. It includes the population balancing term and Pauli string constraints, which
are no longer necessary in a dequantized setting. This code is included for reference, but should not be used for any serious
experimentation. This code includes all-necessary tools to switch to a Y-rotation based variational circuit, 
though it is currently implemented using a product of exponentials of Lie generators.

htaac_qsos_v2.cpp:
This is the second version of the HTAAC-QSOS code. It removes the population balancing term and Pauli string constraints
in favor a simple constraint-based penalty. This code includes all-necessary tools to switch to a Y-rotation based
variational circuit, though it is currently implemented using a product of exponentials of Lie generators.

htaac_qsos_v3.cpp:
This is the third version of the HTAAC-QSOS code. This code removes the fragments that accomodate a Y-rotation based
variational circuit, making it faster and completely dropping the qubit-based variational circuit structure.

