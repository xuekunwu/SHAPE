import os
import json
import time


def save_feedback(cache_dir: str, feedback_type: str, comment: str = None):
    print(f"==> Saving feedback to {cache_dir}")
    print(f"==> Feedback type: {feedback_type}")
    print(f"==> Comment: {comment}")

    feedback_file = os.path.join(cache_dir, "user_feedback.json")
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []
    
    feedback_data.append({
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "feedback_type": feedback_type,
        "comment": comment
    })

    # Save feedback
    with open(feedback_file, 'w') as f:
        json.dump(feedback_data, f, indent=4)
