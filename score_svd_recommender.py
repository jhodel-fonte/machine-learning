from surprise import dump
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def get_top_recommendations(student_id, borrowed_books, top_n=10):
    try:
        _, algo = dump.load(str(BASE_DIR / 'library_svd_model.pkl'))
        with open(BASE_DIR / 'all_books.json', 'r') as f:
            all_books = json.load(f)
    except FileNotFoundError:
        return {"error": "Model files missing. Run training script first."}

    borrowed_books = {str(book).strip() for book in (borrowed_books or [])}
    recommendations = []
    
    # Predict rating only for books the student has not yet read
    for book in all_books:
        if str(book) not in borrowed_books:
            pred = algo.predict(uid=student_id, iid=book)
            recommendations.append({
                "isbn": book, 
                "predicted_score": round(pred.est, 2)
            })
            
    # Sort from highest predicted rating to lowest
    recommendations.sort(key=lambda x: x["predicted_score"], reverse=True)
    
    return recommendations[:max(0, top_n)]