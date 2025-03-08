const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');
const pdfParse = require('pdf-parse');
const dotenv = require('dotenv');

dotenv.config();

// Configuration
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || "AIzaSyAGwF77rylskhbDu4WLNf0zSWTuVlNbr5A";
const EMBEDDING_MODEL = 'embedding-001';
const GENERATION_MODEL = 'gemini-1.5-flash';

// Helper function to split text into manageable chunks
function splitIntoChunks(text, maxChunkSize = 1000) {
  if (!text) return [];
  
  const chunks = [];
  const sentences = text.split(/(?<=[.!?])\s+/);
  let currentChunk = '';

  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length > maxChunkSize && currentChunk.length > 0) {
      chunks.push(currentChunk.trim());
      currentChunk = '';
    }
    currentChunk += sentence + ' ';
  }

  if (currentChunk.trim().length > 0) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

// Function to extract text from PDF
async function extractTextFromPDF(pdfPath) {
  try {
    const dataBuffer = await fs.readFile(pdfPath);
    const data = await pdfParse(dataBuffer);
    return data.text;
  } catch (error) {
    throw new Error(`Failed to extract text from PDF: ${error.message}`);
  }
}

// Function to get embeddings from Google's Generative AI API
async function getEmbedding(content, taskType, title = '') {
  if (!content) throw new Error('Content is required for embedding');
  
  const requestBody = {
    content: { parts: [{ text: content }] },
    taskType: taskType.toUpperCase()
  };

  if (taskType.toUpperCase() === 'RETRIEVAL_DOCUMENT' && title) {
    requestBody.title = title;
  }

  try {
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1beta/models/${EMBEDDING_MODEL}:embedContent`,
      requestBody,
      {
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': GOOGLE_API_KEY
        }
      }
    );

    return response.data.embedding;
  } catch (error) {
    throw new Error(`Failed to get embedding: ${error.response?.data?.error?.message || error.message}`);
  }
}

// Calculate dot product of two vectors
function dotProduct(vecA, vecB) {
  if (!vecA || !vecB || vecA.length !== vecB.length) {
    throw new Error('Invalid vectors for dot product calculation');
  }
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
}

// Function to generate content using Google's Generative AI API
async function generateContent(prompt) {
  const generationConfig = {
    temperature: 0.2,
    maxOutputTokens: 1024,
  };

  try {
    const response = await axios.post(
      `https://generativelanguage.googleapis.com/v1/models/${GENERATION_MODEL}:generateContent`,
      {
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig,
      },
      {
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": GOOGLE_API_KEY,
        }
      }
    );

    const text = response.data.candidates[0].content.parts[0].text;

    // Regex to match JSON content within markdown code blocks
    const jsonRegex = /```json\s*([\s\S]*?)\s*```/;
    const match = text.match(jsonRegex);

    if (match && match[1]) {
      const jsonString = match[1].trim();
      try {
        const parsedJson = JSON.parse(jsonString);
        return parsedJson;
      } catch (e) {
        console.warn('Extracted content is not valid JSON:', jsonString);
        return jsonString;
      }
    } else {
      console.warn('No JSON found in response:', text);
      return text;
    }
  } catch (error) {
    throw new Error(`Failed to generate content: ${error.response?.data?.error?.message || error.message}`);
  }
}

// Main function to process a PDF document and create an embeddings database
async function processPDFDocument(pdfPath, outputJsonPath) {
  try {
    if (!GOOGLE_API_KEY) throw new Error('GOOGLE_API_KEY is not set in environment variables');

    const fullText = await extractTextFromPDF(pdfPath);
    const chunks = splitIntoChunks(fullText);

    const documents = await Promise.all(
      chunks.map(async (chunk, i) => {
        const title = `Chunk ${i + 1}`;
        const embedding = await getEmbedding(chunk, 'retrieval_document', title);
        return { title, content: chunk, embedding };
      })
    );

    await fs.writeFile(outputJsonPath, JSON.stringify(documents, null, 2));

    return { 
      success: true, 
      message: 'PDF processed successfully', 
      documentsCount: documents.length 
    };
  } catch (error) {
    console.error('Error processing PDF document:', error);
    return { success: false, message: error.message };
  }
}

// Function to query the document database
async function queryDocument(query, embeddingsJsonPath) {
  try {
    if (!query) throw new Error("Query cannot be empty");

    const documents = JSON.parse(await fs.readFile(embeddingsJsonPath, "utf8"));
    const queryEmbedding = await getEmbedding(query, "retrieval_query");

    let maxDotProduct = -Infinity;
    let mostRelevantPassage = null;

    for (const doc of documents) {
      const similarity = dotProduct(queryEmbedding.values, doc.embedding.values);
      if (similarity > maxDotProduct) {
        maxDotProduct = similarity;
        mostRelevantPassage = doc.content;
      }
    }

    if (!mostRelevantPassage) {
      return {
        success: false,
        message: "No relevant information found in the document.",
      };
    }

    // Define the desired schema
    const schemaDescription = `{
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "EmployeeName": {
            "type": "string",
            "description": "Designation Of Employee"
          }
        },
        "required": ["EmployeeName"]
      }
    }`;

    // Construct the prompt with schema instructions and an example
    const prompt = `Based on the following passage, extract the employee designations and return them as a JSON string adhering to this schema: ${schemaDescription}

For example, if the passage mentions "John is a Manager and Alice is an Engineer", the output should be:
\`\`\`json
[
  {"EmployeeName": "Manager"},
  {"EmployeeName": "Engineer"}
]
\`\`\`

Passage:
${mostRelevantPassage}

Return the result as a valid JSON string wrapped in \`\`\`json and \`\`\`.`;

    const answer = await generateContent(prompt);

    return {
      success: true,
      query,
      relevantPassage: mostRelevantPassage,
      answer,
      confidence: maxDotProduct,
    };
  } catch (error) {
    console.error("Error querying document:", error);
    return { success: false, message: error.message };
  }
}

// Example usage
async function main() {
  try {
    const processResult = await processPDFDocument(
      './document.pdf',
      './embeddings-database.json'
    );

    if (processResult.success) {
      const queryResult = await queryDocument(
        `who is d.srikanth ?
        Data = {''}
        `,
        './embeddings-database.json'
      );
      console.log(JSON.stringify(queryResult.answer, null, 2));
    }
  } catch (error) {
    console.error('Error in main:', error);
  }
}

module.exports = {
  processPDFDocument,
  queryDocument
};

// Run as standalone script
main();