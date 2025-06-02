import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://0.0.0.0:9090";

export async function POST(request: NextRequest) {
  try {
    // Parse the incoming JSON body

    // const formData = await request.formData();
    // // const prompt = formData.get("text");
    // const textInput = formData.get("text");
    // console.log("textInput");
    // console.log(textInput);

    // if (!textInput) {
    //   throw new Error("No valid text provided");
    // }

    const body = await request.json();
    var { text } = body;
    // var textInput = "check this"
    console.log("textInput-text");
    console.log(text);

    if (!text) {
      return NextResponse.json(
        { error: "Text input is required" },
        { status: 400 }
      );
    }

    const url = new URL(`${BACKEND_URL}/generate_image`);
    console.log(url);
    console.log(JSON.stringify({  text: text }));
    // Forward the text input to the backend

    // const response = await fetch(`${BACKEND_URL}/generate_image`, {
    // method: "POST",
    // headers: {
    //     "Content-Type": "application/json",  // Indicating the content type
    // },
    // body: JSON.stringify({ textInput }),  // Send the 'text' as a JSON object
    // });

    // var resp = document.getElementById("resp");
    // var formData = new FormData();
    // formData.append("text", textInput);
    // formData.append("password", "pass");
    
    // const response = await fetch("http://0.0.0.0:9090/generate_image", {
    //         method: 'POST',
    //         body: formData,
    //     })
    
    // Make the request to your FastAPI server, passing 'text' as a query parameter
    const response = await fetch(`${BACKEND_URL}/generate_image?text=${encodeURIComponent(text)}`, {
    method: "POST",
    });


    // const response = await fetch(url, {
    //   method: "POST",
    //   headers: {
    //     "Content-Type": "application/json",
    //   },
    //   body: JSON.stringify({  textInput }),
    // });
    console.log(response);

    if (!response.ok) {
      throw new Error(`Backend error: ${response.statusText}`);
    }

    // Get the image data (as binary) from the backend
    const imageBuffer = await response.arrayBuffer();

    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": "image/png",
      },
    });
  } catch (error) {
    console.error("Error processing text input:", error);
    return NextResponse.json(
      { error: "Failed to process the input" },
      { status: 500 }
    );
  }
}
