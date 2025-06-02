"use client";
import { useState } from 'react';

const SimpleQuery = () => {
  const [textInput, setTextInput] = useState("");  // For text input
  const [prediction, setPrediction] = useState(null); // For prediction result
  const [loading, setLoading] = useState(false); // Loading state for the API request
  const [error, setError] = useState(null); // Error state

  // Handle text input change
  const handleTextInputChange = (e) => {
    setTextInput(e.target.value);
    setError(null); // Clear any previous errors
  };

  // Handle the form submission
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent the default form behavior

    if (!textInput) {
      setError("Please enter some text");
      return;
    }

    setLoading(true);
    setError(null); // Clear any previous errors

    try {
      const response = await fetch("/generate", {
        method: "POST", // Use POST for submitting the text input
        headers: {
          "Content-Type": "application/json", // If you're sending JSON data
        },
        body: JSON.stringify({ text: textInput }), // Pass the text input as a JSON body
      });

      if (!response.ok) {
        throw new Error("Failed to fetch image from the API");
      }

      // Get the image as a blob
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setPrediction(imageUrl); // Update the state with the image URL
    } catch (err) {
      setError(err.message); // Handle error
    } finally {
      setLoading(false); // Reset loading state
    }
  };

  return (
    <div className="max-w-md mx-auto mt-8 p-6 border rounded-lg shadow-sm">
      <h2 className="text-xl font-bold mb-4">Text to Image API</h2>
      
      <div className="space-y-4">
        <div>
          <input
            type="text"
            value={textInput}
            onChange={handleTextInputChange}
            placeholder="Enter text for the image"
            className="mb-4 w-full px-4 py-2 border border-gray-300 rounded-md"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 disabled:bg-gray-400"
        >
          {loading ? 'Processing...' : 'Generate Image'}
        </button>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {prediction && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-medium mb-2">Generated Image:</h3>
            <img src={prediction} alt="Generated" className="max-w-full rounded" />
          </div>
        )}
      </div>
    </div>
  );
};

export default SimpleQuery;
