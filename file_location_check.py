import os.path

# Check for /resources check for vectors_gl.npy and vocab_gl.npy

catch_exception = False
if not os.path.isfile('resources/vectors_gl.npy'):
    print("vectors_gl.npy file not found in resources. Please download the file and add it to the resources folder.")
    catch_exception = True


if not os.path.isfile('resources/vocab_gl.pickle'):
    print("vocab_gl.pickle file not found in resources. Please download the file and add it to the resources folder.")
    catch_exception = True

if not os.path.isfile('data/data/common/rdf_type_lookup.json'):
    print("rdf_type_lookup.json not found in data/data/common. Please download the file and add it to the "
          "data/data/common folder.")
    catch_exception = True

if not os.path.isfile('data/data/common/relations.pickle'):
    print("relations.pickle not found in data/data/common. Please download the file and add it to the "
          "data/data/common folder.")
    catch_exception = True

if not os.path.isfile('data/data/lcquad/id_big_data.json'):
    print("id_big_data.json file not found in data/data/lcquad. Please download the file and "
          "add it to the data/data/lcquad folder to perform experiments on lcquad.")
    catch_exception = True

if not os.path.isfile('data/data/qald/id_big_data.json'):
    print("id_big_data.json file not found in data/data/qald. Please download the file and "
          "add it to the data/data/lcquad folder to perform experiments on qald.")
    catch_exception = True

if not catch_exception:
    print("Found all required files")