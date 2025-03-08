const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const pdfParse = require('pdf-parse');
const dotenv = require('dotenv');

dotenv.config();

// Initialize Express app
const app = express();
const port = process.env.PORT || 3000;

// Configure middleware
app.use(bodyParser.json());

// Set up Gemini API
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "YOUR_API_KEY";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });

// Path to the PDF and embeddings file
const PDF_PATH = path.join(__dirname, 'merged_pdf.pdf');
const EMBEDDINGS_PATH = path.join(__dirname, 'embeddings-database.json');
const ITEM_TO_HS_PATH = path.join(__dirname, 'item-to-hs-mapping.json');

// Function to extract HS Codes from PDF content
function extractHSCodes(pdfText) {
  const hsCodeRegex = /(\d{8})\s+(.*?)(?:\s+)(Free|Restricted|Prohibited|Not Permitted)/gi;
  const hsCodesData = {};
  const itemToHsMap = {};
  
  let match;
  while ((match = hsCodeRegex.exec(pdfText)) !== null) {
    const hsCode = match[1];
    const description = match[2].trim();
    const exportPolicy = match[3];
    
    hsCodesData[hsCode] = {
      description,
      policy: exportPolicy
    };
    
    // Create a mapping from items in description to HS code
    const items = description.split(/[,;\/]/).map(item => item.trim().toLowerCase());
    items.forEach(item => {
      if (item.length > 3) { // Ignore very short terms
        itemToHsMap[item] = hsCode;
      }
    });
    
    // Also add the full description as a searchable item
    itemToHsMap[description.toLowerCase()] = hsCode;
  }
  
  return { hsCodesData, itemToHsMap };
}

// Function to generate embeddings and extract HS codes from PDF
async function generatePdfEmbeddings() {
  try {
    if (fs.existsSync(EMBEDDINGS_PATH) && fs.existsSync(ITEM_TO_HS_PATH)) {
      console.log('Embeddings files already exist. Using existing data.');
      const embeddingsData = JSON.parse(fs.readFileSync(EMBEDDINGS_PATH, 'utf8'));
      const itemToHsMap = JSON.parse(fs.readFileSync(ITEM_TO_HS_PATH, 'utf8'));
      return { ...embeddingsData, itemToHsMap };
    }

    console.log('Generating embeddings from PDF...');
    
    const dataBuffer = fs.readFileSync(PDF_PATH);
    const pdfData = await pdfParse(dataBuffer);
    const pdfText = pdfData.text;
    
    console.log('Extracted PDF Text length:', pdfText.length); 
    
    const { hsCodesData, itemToHsMap } = extractHSCodes(pdfText);
    
    // Save item to HS code mapping
    fs.writeFileSync(ITEM_TO_HS_PATH, JSON.stringify(itemToHsMap, null, 2));
    console.log('Item to HS code mapping saved to file.');
    
    const chunkSize = 1000;
    const textChunks = [];
    
    const paragraphs = pdfText.split('\n\n');
    let currentChunk = '';
    
    for (const paragraph of paragraphs) {
      if ((currentChunk + paragraph).length > chunkSize) {
        if (currentChunk.length > 0) {
          textChunks.push(currentChunk.trim());
          currentChunk = '';
        }
        
        if (paragraph.length > chunkSize) {
          const words = paragraph.split(' ');
          let subChunk = '';
          
          for (const word of words) {
            if ((subChunk + ' ' + word).length > chunkSize) {
              textChunks.push(subChunk.trim());
              subChunk = word;
            } else {
              subChunk += ' ' + word;
            }
          }
          
          if (subChunk.length > 0) {
            currentChunk = subChunk.trim();
          }
        } else {
          currentChunk = paragraph;
        }
      } else {
        currentChunk += '\n\n' + paragraph;
      }
    }
    
    if (currentChunk.length > 0) {
      textChunks.push(currentChunk.trim());
    }
    
    const embeddingsDatabase = {
      chunks: [],
      hsCodesData: hsCodesData
    };
    
    for (let i = 0; i < textChunks.length; i++) {
      const chunk = textChunks[i];
      const embedResult = await embeddingModel.embedContent({
        content: { parts: [{ text: chunk }] },
      });
      
      const embedding = embedResult.embedding.values;
      
      embeddingsDatabase.chunks.push({
        id: i,
        content: chunk,
        embedding: embedding
      });
      
      console.log(`Generated embedding for chunk ${i + 1}/${textChunks.length}`);
    }
    
    fs.writeFileSync(EMBEDDINGS_PATH, JSON.stringify(embeddingsDatabase, null, 2));
    console.log('Embeddings saved to file.');
    
    return { ...embeddingsDatabase, itemToHsMap };
  } catch (error) {
    console.error('Error generating embeddings:', error);
    throw error;
  }
}

// Function to find relevant content using embeddings
async function findRelevantContent(query, embeddingsDatabase) {
  try {
    const queryEmbedResult = await embeddingModel.embedContent({
      content: { parts: [{ text: query }] },
    });
    
    const queryEmbedding = queryEmbedResult.embedding.values;
    
    const similarityScores = embeddingsDatabase.chunks.map(item => {
      const similarity = cosineSimilarity(queryEmbedding, item.embedding);
      return { ...item, similarity };
    });
    
    const topResults = similarityScores
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);
    
    return topResults.map(item => item.content).join('\n\n');
  } catch (error) {
    console.error('Error finding relevant content:', error);
    throw error;
  }
}

// Function to find HS code by item name
function findHSCodeByItemName(itemName, itemToHsMap) {
  const normalizedItemName = itemName.toLowerCase().trim();
  
  // Direct match
  if (itemToHsMap[normalizedItemName]) {
    return itemToHsMap[normalizedItemName];
  }
  
  // Partial match
  const itemKeys = Object.keys(itemToHsMap);
  
  // Check if item name is contained in any key
  const containsMatch = itemKeys.find(key => key.includes(normalizedItemName));
  if (containsMatch) {
    return itemToHsMap[containsMatch];
  }
  
  // Check if any key is contained in item name
  const isContainedMatch = itemKeys.find(key => normalizedItemName.includes(key) && key.length > 5);
  if (isContainedMatch) {
    return itemToHsMap[isContainedMatch];
  }
  
  return null;
}

async function checkHSCodeCompliance(hsCode, embeddingsDatabase) {
  if (embeddingsDatabase.hsCodesData && embeddingsDatabase.hsCodesData[hsCode]) {
    const hsData = embeddingsDatabase.hsCodesData[hsCode];
    return {
      exists: true,
      allowed: hsData.policy.toLowerCase() === 'free',
      policy: hsData.policy,
      description: hsData.description
    };
  }
  
  if (hsCode.length >= 4) {
    const chapter = hsCode.substring(0, 4);
    const twoDigitChapter = hsCode.substring(0, 2);
    
    const matchingCodes = Object.keys(embeddingsDatabase.hsCodesData || {})
      .filter(code => code.startsWith(chapter) || code.startsWith(twoDigitChapter));
    
    if (matchingCodes.length > 0) {
      const policies = matchingCodes.map(code => embeddingsDatabase.hsCodesData[code].policy);
      
      if (policies.some(policy => policy.toLowerCase() === 'free')) {
        return {
          exists: true,
          allowed: true,
          policy: 'Free',
          description: `Falls under chapter ${chapter} which has some free categories`
        };
      } else {
        return {
          exists: true,
          allowed: false,
          policy: policies[0],
          description: `Falls under chapter ${chapter} which has no free categories`
        };
      }
    }
  }
  
  try {
    // Use Gemini to generate a reason
    const prompt = `Given HS code ${hsCode} that wasn't found in our database, provide a reason why this code might not be recognized. Limit your response to one short paragraph.`;
    
    const result = await model.generateContent({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 100 }
    });
    
    const dynamicReason = result.response.text();
    
    return {
      exists: false,
      allowed: false,
      reason: dynamicReason
    };
  } catch (error) {
    console.error('Error generating dynamic reason:', error);
    return {
      exists: false,
      allowed: false,
      reason: `The HS Code ${hsCode} was not found in the export compliance regulations. Please verify the code and try again.`
    };
  }
}

// Helper function to calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  
  return dotProduct / (normA * normB);
}

// Initialize embeddings database
let embeddingsDatabase = {
  chunks: [],
  hsCodesData: {},
  itemToHsMap: {}
};

// API endpoint to check export compliance
app.post('/api/check-export-compliance', async (req, res) => {
  try {
    const { hsCode, itemWeight, material, itemName, itemManufacturer } = req.body;
    
    if (!hsCode && !itemName) {
      return res.status(400).json({
        status: false,
        error: "Missing required fields. Please provide either hsCode or itemName"
      });
    }
    
    let codeToCheck = hsCode;
    
    // If hsCode is not provided but itemName is, try to find the HS code
    if (!hsCode && itemName) {
      codeToCheck = findHSCodeByItemName(itemName, embeddingsDatabase.itemToHsMap);
      
      if (!codeToCheck) {
        return res.json({
          status: false,
          allowed: false,
          reason: `Could not find an HS code matching item name: ${itemName}. Please provide a valid HS code.`
        });
      }
    }
    
    const hsCodeCompliance = await checkHSCodeCompliance(codeToCheck, embeddingsDatabase);
    
    if (!hsCodeCompliance.exists) {
      return res.json({
        status: false,
        allowed: false,
        reason: hsCodeCompliance.reason,
        queriedHsCode: codeToCheck,
        queriedItemName: itemName || null
      });
    }
    
    if (hsCodeCompliance.allowed) {
      return res.json({
        status: true,
        allowed: true,
        hsCode: codeToCheck,
        policy: hsCodeCompliance.policy,
        description: hsCodeCompliance.description,
        conditions: "Standard export conditions apply",
        queriedItemName: itemName || null
      });
    } else {
      return res.json({
        status: false,
        allowed: false,
        hsCode: codeToCheck,
        policy: hsCodeCompliance.policy,
        description: hsCodeCompliance.description,
        reason: `Export not allowed for HS Code ${codeToCheck} with policy ${hsCodeCompliance.policy}`,
        queriedItemName: itemName || null
      });
    }
    
  } catch (error) {
    console.error('Error checking export compliance:', error);
    return res.status(500).json({
      status: false,
      error: "An error occurred while checking export compliance"
    });
  }
});

app.post('/api/find-by-description', (req, res) => {
    try {
      const { description } = req.body;
      
      if (!description) {
        return res.status(400).json({
          status: false,
          error: "Missing required field: description"
        });
      }
      
      const normalizedDescription = description.toLowerCase().trim();
      
      // Search through the HS codes data
      const hsCodesData = embeddingsDatabase.hsCodesData || {};
      const matchingHsCode = Object.keys(hsCodesData).find(hsCode => 
        hsCodesData[hsCode].description.toLowerCase() === normalizedDescription
      );
      
      if (matchingHsCode) {
        return res.json({
          hsCode: matchingHsCode
        });
      }
      
      // If no exact match, try partial match
      const partialMatchHsCode = Object.keys(hsCodesData).find(hsCode => 
        hsCodesData[hsCode].description.toLowerCase().includes(normalizedDescription) ||
        normalizedDescription.includes(hsCodesData[hsCode].description.toLowerCase())
      );
      
      if (partialMatchHsCode) {
        return res.json({
          status: true,
          hsCode: partialMatchHsCode,
          note: "Found via partial match"
        });
      }
      
      return res.json({
        status: false,
        error: "No matching HS code found for this description"
      });
      
    } catch (error) {
      console.error('Error finding HS code by description:', error);
      return res.status(500).json({
        status: false,
        error: "An error occurred while finding HS code"
      });
    }
  });

// API endpoint to get all HS codes in the database
app.get('/api/hs-codes', (req, res) => {
  try {
    if (!embeddingsDatabase.hsCodesData) {
      return res.status(404).json({
        status: false,
        error: "HS Codes data not found. Please regenerate embeddings."
      });
    }
    
    return res.json({
      status: true,
      count: Object.keys(embeddingsDatabase.hsCodesData).length,
      hsCodes: embeddingsDatabase.hsCodesData
    });
  } catch (error) {
    console.error('Error retrieving HS codes:', error);
    return res.status(500).json({
      status: false,
      error: "An error occurred while retrieving HS codes"
    });
  }
});

// API endpoint to get item to HS code mapping
app.get('/api/item-to-hs-mapping', (req, res) => {
  try {
    if (!embeddingsDatabase.itemToHsMap) {
      return res.status(404).json({
        status: false,
        error: "Item to HS code mapping not found. Please regenerate embeddings."
      });
    }
    
    return res.json({
      status: true,
      count: Object.keys(embeddingsDatabase.itemToHsMap).length,
      mapping: embeddingsDatabase.itemToHsMap
    });
  } catch (error) {
    console.error('Error retrieving item to HS mapping:', error);
    return res.status(500).json({
      status: false,
      error: "An error occurred while retrieving item to HS mapping"
    });
  }
});

// Endpoint to force regeneration of embeddings
app.post('/api/regenerate-embeddings', async (req, res) => {
  try {
    if (fs.existsSync(EMBEDDINGS_PATH)) {
      fs.unlinkSync(EMBEDDINGS_PATH);
    }
    
    if (fs.existsSync(ITEM_TO_HS_PATH)) {
      fs.unlinkSync(ITEM_TO_HS_PATH);
    }
    
    embeddingsDatabase = await generatePdfEmbeddings();
    
    res.json({
      success: true,
      message: "Embeddings and item mapping regenerated successfully",
      chunksCount: embeddingsDatabase.chunks.length,
      hsCodesCount: Object.keys(embeddingsDatabase.hsCodesData || {}).length,
      itemMappingsCount: Object.keys(embeddingsDatabase.itemToHsMap || {}).length
    });
  } catch (error) {
    console.error('Error regenerating embeddings:', error);
    res.status(500).json({
      success: false,
      error: "Failed to regenerate embeddings"
    });
  }
});

// Initialize server
async function initServer() {
  try {
    embeddingsDatabase = await generatePdfEmbeddings();
    
    app.listen(port, () => {
      console.log(`Export compliance API server running on port ${port}`);
      console.log(`Loaded ${embeddingsDatabase.chunks?.length || 0} embedded chunks from PDF`);
      console.log(`Extracted ${Object.keys(embeddingsDatabase.hsCodesData || {}).length} HS codes from PDF`);
      console.log(`Created ${Object.keys(embeddingsDatabase.itemToHsMap || {}).length} item to HS code mappings`);
    });
  } catch (error) {
    console.error('Failed to initialize server:', error);
    process.exit(1);
  }
}

// Start the server
initServer();