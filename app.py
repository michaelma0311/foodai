"""
Localhost model using Flask that provdes web interface
"""

from flask import Flask, render_template, request, jsonify
from rag_modal_query import ModalRecipeRAG
import json
from google import genai

app = Flask(__name__)

# Global RAG instance
rag_system = None

def init_rag():
    """Intiailize RAG"""
    print("Initializing RAG system")
    global rag_system
    if rag_system is None:
        rag_system = ModalRecipeRAG()
        if not rag_system.load_index():
            print("Failed to load RAG")
            return False
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.before_request
def preload_rag():
    """Preload RAG ."""
    init_rag()


@app.route('/api/search', methods=['POST'])
def search_recipes():
    """API endpoint"""
    if rag_system is None or not rag_system.is_loaded:
        return jsonify({'error': 'RAG system not available'}), 500

    try:
        data = request.get_json()
        ingredients = data.get('ingredients', [])
        top_k = data.get('top_k', 5)

        if not ingredients:
            return jsonify({'error': 'No ingredients provided'}), 400

        # Build query
        query = f"I have these ingredients: {', '.join(ingredients)}. What recipes can I make?"
        results = rag_system.search_recipes(query, top_k=top_k)

        # Format results for frontend
        formatted_results = []
        for recipe in results:
            formatted_results.append({
                'title': recipe['title'],
                'similarity_score': round(recipe['similarity_score'], 4),
                'ingredients': recipe['ingredients'],
                'link': recipe['link'],
                'source': recipe['source']
            })

        return jsonify({
            'query': query,
            'results': formatted_results,
            'total_results': len(formatted_results)
        })

    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/substitutions', methods=['POST'])
def get_substitutions():
    data = request.get_json()
    ingredients = data.get('ingredients', [])

    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400

    try:
        client = genai.Client(api_key="AIzaSyAinIyitInaxHK3fsZpgN-hEcyJgIQe4jU")

        prompt = f"""The user has these ingredients: {', '.join(ingredients)}.
For each ingredient, suggest the top 2 common cooking substitutions.
Respond ONLY with a valid JSON object in this exact format:
{{
  "ingredient_name": ["substitution1", "substitution2"]
}}
Only include ingredients that have useful substitutions. No markdown, no explanation."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        response_text = response.text.strip()
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]

        substitutions = json.loads(response_text.strip())
        expanded = list(ingredients)
        for subs in substitutions.values():
            expanded.extend(subs)

        recipes = []
        if rag_system and rag_system.is_loaded:
            query = f"I have these ingredients: {', '.join(expanded)}. What recipes can I make?"
            recipes = rag_system.search_recipes(query, top_k=8)

        return jsonify({'substitutions': substitutions, 'recipes': recipes})

    except Exception as e:
        print(f"Substitution error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    """Get system status."""
    if not init_rag():
        return jsonify({
            'status': 'error',
            'message': 'RAG system not loaded'
        })

    try:
        # Get some stats
        cursor = rag_system.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM recipes")
        recipe_count = cursor.fetchone()[0]

        return jsonify({
            'status': 'ready',
            'recipe_count': recipe_count,
            'index_loaded': rag_system.is_loaded
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    print("Starting Web App")
    print("Initializing RAG")
    if init_rag():
        print("Ready")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize RAG.")