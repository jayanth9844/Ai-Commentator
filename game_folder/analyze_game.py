import chess
import chess.engine
import json

STOCKFISH_PATH = r"C:\Users\Jayanth Raj G\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

class ChessAnalyzer:
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.previous_score = 0
        self.move_number = 0  # Start from 1 instead of 0
        
    def get_phase(self, move_number):
        if move_number <= 10:
            return "Opening"
        elif move_number <= 30:
            return "Middlegame"
        else:
            return "Endgame"

    def classify_move(self, score_cp, prev_score_cp):
        if score_cp is None or prev_score_cp is None:
            return "Unknown"
            
        # Adjust the perspective based on who's moving
        score_diff = prev_score_cp - score_cp
        
        if abs(score_diff) < 20:
            return "Excellent move"
        elif abs(score_diff) < 50:
            return "Good move"
        elif abs(score_diff) < 100:
            return "Inaccuracy"
        elif abs(score_diff) < 300:
            return "Mistake"
        else:
            return "Blunder"

    def get_move_type(self, board, move):
        if board.is_capture(move):
            return "capture"
        elif board.is_castling(move):
            return "castling"
        elif board.is_en_passant(move):
            return "en_passant"
        elif move.promotion is not None:
            return "promotion"
        else:
            return "normal"

    def get_captured_piece(self, board, move):
        if board.is_capture(move):
            if board.is_en_passant(move):
                return "p"
            captured_square = move.to_square
            piece = board.piece_at(captured_square)
            if piece:
                return piece.symbol().lower()
        return ""

    def analyze_position(self, board, last_move=None, depth=20):
        try:
            if last_move:
                # Increment move number for every move, regardless of color
                self.move_number += 1

            # Get previous position score before the move
            prev_score = self.previous_score

            # Analyze current position
            info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            
            # Calculate score in centipawns
            current_score = score.score() / 100 if score.score() is not None else None
            
            # Prepare evaluation string
            if score.is_mate():
                evaluation = f"Mate in {abs(score.mate())}"
                mate_threat = f"Mate in {abs(score.mate())} moves"
            else:
                evaluation = f"{current_score:.2f} pawns advantage to {'White' if current_score >= 0 else 'Black'}"
                mate_threat = "No immediate mate threat detected"
            
            # Calculate score change
            if current_score is not None and prev_score is not None:
                score_change = abs(current_score - prev_score)
                score_change_str = f"The evaluation {'increased' if current_score > prev_score else 'decreased'} by {score_change:.2f} pawns"
            else:
                score_change_str = "Unable to calculate score change"

            # Get best line
            best_line = info.get("pv", [])[:5]  # Get first 5 moves of best line
            best_moves = []
            for m in best_line:
                try:
                    if m in board.legal_moves:
                        best_moves.append(board.san(m))
                except ValueError:
                    best_moves.append(m.uci())
            best_line_str = ", ".join(best_moves)

            # Update previous score for next analysis
            self.previous_score = current_score

            # Prepare move type and capture information if last_move is provided
            move_type = "normal"
            captured = ""
            move_san = ""
            
            if last_move:
                # Safely get move type and capture information
                try:
                    if last_move in board.legal_moves:
                        move_type = self.get_move_type(board, last_move)
                        captured = self.get_captured_piece(board, last_move)
                        move_san = board.san(last_move)
                    else:
                        move_san = last_move.uci()
                except (ValueError, AttributeError):
                    move_san = last_move.uci() if hasattr(last_move, 'uci') else str(last_move)
            
            analysis = {
                "MoveNumber": self.move_number,
                "Move": move_san,
                "CurrentPlayer": "White" if board.turn == chess.BLACK else "Black",  # The player who just moved
                "Phase": self.get_phase(self.move_number),
                "Evaluation": evaluation,
                "ScoreChange": score_change_str,
                "Classification": self.classify_move(current_score, prev_score) if last_move else "Starting position",
                "BestLine": best_line_str,
                "MateThreat": mate_threat,
                "Depth": depth,
                "Captured": captured,
                "MoveType": move_type,
                "Check": "Yes" if board.is_check() else "No",
                "Checkmate": "Yes" if board.is_checkmate() else "No",
                "GameOver": "Yes" if board.is_game_over() else "No",
                "GameOverReason": self.get_game_over_reason(board),
                "Winner": self.get_winner(board)
            }
            
            return analysis

        except Exception as e:
            print(f"Analysis error: {e}")
            return None

    def get_game_over_reason(self, board):
        if board.is_checkmate():
            return "Checkmate"
        elif board.is_stalemate():
            return "Stalemate"
        elif board.is_insufficient_material():
            return "Insufficient material"
        elif board.is_fivefold_repetition():
            return "Fivefold repetition"
        elif board.is_fifty_moves():
            return "Fifty-move rule"
        else:
            return "Game continues"

    def get_winner(self, board):
        if board.is_checkmate():
            return "Black" if board.turn == chess.WHITE else "White"
        return "None"

    def close(self):
        if self.engine:
            self.engine.quit()

def main():
    board = chess.Board()
    analyzer = ChessAnalyzer()
    
    # Example PGN
    moves = ["e4", "e5", "Nf3", "d6"]
    for move in moves:
        move_obj = board.push_san(move)
        analysis = analyzer.analyze_position(board, move_obj)
        print(f"After {move}:")
        print(json.dumps(analysis, indent=2))
    
    analyzer.close()

if __name__ == "__main__":
    main()
