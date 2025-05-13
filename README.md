# How to Run the Project

Follow these steps to set up and run the audio guidance system for visually impaired individuals:

## Step 1: Install Node Modules
Make sure you have Node.js installed on your system. Then, navigate to the project directory and run:

```bash
npm install
```

## Step 2: Install Python Dependencies
Install the required Python packages listed in the `requirements.txt` file. Run the following command:

```bash
pip install -r requirements.txt
```

## Step 3: Run `convert_to_onnx`
Execute the script to convert the model to ONNX format:

```bash
python convert_to_onnx.py
```

## Step 4: Start Ngrok on Local Port 5000
Run Ngrok to expose your local server to the internet:

```bash
ngrok http 5000
```

Note down the Ngrok public URL that is displayed. It will look something like this:

```
https://8c92-2407-c00-xxxx-b61b-xxxx-6e-8a5b-a9f4.ngrok-free.app
```

## Step 5: Update the URL in `client.html`
Open the `client.html` file and update the URL to the Ngrok public URL you obtained in the previous step.

Replace the placeholder URL (in line 285) with your Ngrok URL:

```html
<script>
    const response = await fetch('<<replace this part with link>>/upload_frame',
</script>
```

Save the changes.

## Step 6: Run the Server
Finally, start the Python server:

```bash
python Server.py
```

open the NgRok app link on any device and use !!!

Your system is now up and running. The client should be able to interact with the server using the updated Ngrok URL.
