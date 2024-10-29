import React, { useState } from 'react';
import { View, Text, Button, Image, StyleSheet, Alert } from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
import axios from 'axios';

const App = () => {
  const [imageUri, setImageUri] = useState(null);
  const [prediction, setPrediction] = useState(null);

  // Function to pick image from the camera
  const pickImageFromCamera = async () => {
    ImagePicker.launchCamera(
      {
        mediaType: 'photo',
        includeBase64: true,
        quality: 1,
      },
      (response) => {
        if (response.didCancel) {
          console.log('User cancelled image picker');
        } else if (response.error) {
          console.log('ImagePicker Error: ', response.error);
        } else {
          const uri = response.assets[0].uri;
          setImageUri(uri);
          handlePrediction(response.assets[0]);
        }
      }
    );
  };

  // Function to pick image from the gallery
  const pickImageFromGallery = async () => {
    ImagePicker.launchImageLibrary(
      {
        mediaType: 'photo',
        includeBase64: true,
        quality: 1,
      },
      (response) => {
        if (response.didCancel) {
          console.log('User cancelled image picker');
        } else if (response.error) {
          console.log('ImagePicker Error: ', response.error);
        } else {
          const uri = response.assets[0].uri;
          setImageUri(uri);
          handlePrediction(response.assets[0]);
        }
      }
    );
  };

  // Function to send image to FastAPI backend
  const handlePrediction = async (image) => {
    const formData = new FormData();
    formData.append('file', {
      uri: image.uri,
      type: image.type,
      name: image.fileName || 'uploaded_image.jpg',
    });

    try {
      const response = await axios.post('http://10.0.2.2:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (error) {
      console.error('Error uploading image:', error);
      Alert.alert('Error', 'Failed to upload image');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Leaf Disease Prediction</Text>

      {imageUri && (
        <Image source={{ uri: imageUri }} style={styles.image} />
      )}

      {prediction && (
        <View style={styles.prediction}>
          <Text style={styles.predictionText}>
            Disease: {prediction.class}
          </Text>
          <Text style={styles.predictionText}>
            Confidence: {(prediction.confidence * 100).toFixed(2)}%
          </Text>
        </View>
      )}

      <View style={styles.buttonContainer}>
        <Button title="Capture Image" onPress={pickImageFromCamera} />
        <Button title="Upload Image" onPress={pickImageFromGallery} />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  image: {
    width: 300,
    height: 300,
    borderRadius: 10,
    marginBottom: 20,
  },
  prediction: {
    marginBottom: 20,
    alignItems: 'center',
  },
  predictionText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
  },
});

export default App;
