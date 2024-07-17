from pyrouge import Rouge155
r = Rouge155()
 
r.system_dir = './candidate/'
r.model_dir = './reference/'
r.system_filename_pattern = '(\d+)_candidate.txt'
r.model_filename_pattern = '#ID#_reference.txt'
 
output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
