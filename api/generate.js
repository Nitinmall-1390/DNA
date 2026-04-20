// api/generate.js
// This file runs on Vercel's server - the API key is NEVER sent to the browser.
// Place this file at:  your-project/api/generate.js

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  // Read your secret API key from Vercel environment variables
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({
      error: "ANTHROPIC_API_KEY environment variable is not set on Vercel.",
    });
  }

  try {
    const { prompt } = req.body;

    if (!prompt || typeof prompt !== "string") {
      return res.status(400).json({ error: "Missing or invalid prompt in request body." });
    }

    // Call Anthropic from the server — no CORS issues here
    const anthropicResponse = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    const data = await anthropicResponse.json();

    if (!anthropicResponse.ok) {
      return res.status(anthropicResponse.status).json({
        error: data?.error?.message || "Anthropic API returned an error.",
        details: data,
      });
    }

    // Return the full Anthropic response to the frontend
    return res.status(200).json(data);
  } catch (err) {
    return res.status(500).json({ error: `Server error: ${err.message}` });
  }
}
