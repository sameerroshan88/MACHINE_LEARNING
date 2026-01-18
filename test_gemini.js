const { GoogleGenerativeAI } = require('./eeg-alzheimer-blog/node_modules/@google/generative-ai');

async function testModels() {
  // Load API key from environment variable
  const apiKey = process.env.GEMINI_API_KEY;
  
  if (!apiKey) {
    console.error('❌ ERROR: GEMINI_API_KEY not found in environment variables.');
    console.log('Please set it using: export GEMINI_API_KEY="your_api_key_here"');
    console.log('Or create a .env file with: GEMINI_API_KEY=your_api_key_here');
    process.exit(1);
  }
  
  const genAI = new GoogleGenerativeAI(apiKey);
  
  // List of models to try
  const modelsToTry = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-exp', 
    'gemini-1.5-flash-latest',
    'gemini-1.5-pro-latest',
    'gemini-pro',
  ];
  
  console.log('Testing available Gemini models...\n');
  
  for (const modelName of modelsToTry) {
    try {
      console.log(`Testing ${modelName}...`);
      const model = genAI.getGenerativeModel({ model: modelName });
      const result = await model.generateContent('Say hello in 5 words or less');
      const response = await result.response;
      console.log(`✅ ${modelName} WORKS!`);
      console.log(`   Response: ${response.text().trim()}\n`);
      return modelName; // Return first working model
    } catch (error) {
      console.log(`❌ ${modelName} failed: ${error.status || error.message}\n`);
    }
  }
}

testModels().then(working => {
  if (working) console.log(`\n🎉 Use model: ${working}`);
});
