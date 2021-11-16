# define functions 

# load model

# start_microphone

# predict model

# voting Classifier

# two threads -- Recording microphone and load/predict model

# maybe another thread is needed for voting classifier

# start_microphone begins recording in chunks of 1024 bytes
# each chunck is preprocessed and sent to model for prediction
# voting classifier continuosly takes prediction and updatses final result
# process continues until user stops recording or passess the 120 second point and the voting classifier is greater than 95%

