import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

URI = 'http://127.0.0.1:5000'
st.title('MNIST Neural Network Visualizer')
st.sidebar.markdown('## Input Image')
st.set_option('deprecation.showPyplotGlobalUse', False)

if st.button('Get random prediction'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))

    
    st.sidebar.image(image, width=150)
    
    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32, 4))
        if layer == 2:
            row = 2
            col = 5
        else:
            row = 2
            col = 16
        
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number * np.ones([8, 8, 3]).astype('float64'))
            plt.xticks([])
            plt.yticks([])
            
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)

        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.subheader('Layer{}'.format(layer+1))

        st.text(numbers)
        st.pyplot()

        if layer == 2:
            st.text('max is: ')
            st.text(max(numbers))

            predict_index = np.argmax(numbers)
            st.text('The prediction is: ')
            st.subheader(predict_index)
        
                
            
    


