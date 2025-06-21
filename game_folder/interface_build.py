from pygame import gfxdraw
import pygame
import chess
import os
import threading
import queue
import pyttsx3
import torch
from analyze_game import ChessAnalyzer
from gen_commentary import GPTModel, NEW_CONFIG, tokenizer, generate, device, format_input

# Initialize pygame with better graphics
pygame.init()
clock = pygame.time.Clock()

# Enhanced Constants
WIDTH, HEIGHT = 600, 600  # Reduced from 800x800
BOARD_SIZE = 8
SQUARE_SIZE = WIDTH // BOARD_SIZE
FPS = 60  # Increased for smoother animations
BORDER_SIZE = 40  # Reduced border size
ANALYSIS_WIDTH = 400  # Keep analysis width the same

# Enhanced Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (238, 238, 210)  # Softer light square
DARK_SQUARE = (118, 150, 86)    # Softer dark square
HIGHLIGHT = (186, 202, 43, 150)  # Semi-transparent highlight
MOVE_HIGHLIGHT = (106, 135, 77, 150)  # Semi-transparent move indicator
BORDER_COLOR = (62, 66, 73)
TEXT_COLOR = (200, 200, 200)

# Load piece images with enhanced error handling
def load_piece_images():
    pieces = {}
    piece_symbols = ['p', 'r', 'n', 'b', 'q', 'k']
    colors = ['b', 'w']
    
    for color in colors:
        for piece in piece_symbols:
            key = color + piece
            filename = f"chess_pieces/{key}.png"
            
            try:
                if os.path.exists(filename):
                    img = pygame.image.load(filename)
                else:
                    # Create an elegant placeholder piece
                    img = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    center = (SQUARE_SIZE//2, SQUARE_SIZE//2)
                    
                    # Draw piece background
                    if color == 'w':
                        pygame.draw.circle(img, (240, 240, 240, 220), center, SQUARE_SIZE//2.5)
                        text_color = (60, 60, 60)
                    else:
                        pygame.draw.circle(img, (60, 60, 60, 220), center, SQUARE_SIZE//2.5)
                        text_color = (240, 240, 240)
                    
                    # Add piece letter
                    font = pygame.font.SysFont('Arial', SQUARE_SIZE//2, True)
                    text = font.render(piece.upper(), True, text_color)
                    text_rect = text.get_rect(center=center)
                    img.blit(text, text_rect)
                
                pieces[key] = pygame.transform.scale(img, (SQUARE_SIZE-10, SQUARE_SIZE-10))
            except Exception as e:
                print(f"Error loading piece {key}: {e}")
                
    return pieces

# Draw the enhanced chess board
def draw_board(screen):
    # Draw main border
    pygame.draw.rect(screen, BORDER_COLOR, 
                    pygame.Rect(BORDER_SIZE-10, BORDER_SIZE-10, 
                              WIDTH+20, HEIGHT+20))
    
    # Draw squares with coordinates
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x = col * SQUARE_SIZE + BORDER_SIZE
            y = row * SQUARE_SIZE + BORDER_SIZE
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE))
            
            # Draw coordinates
            if col == 0:  # Rank numbers
                font = pygame.font.SysFont('Arial', 20, True)
                text = font.render(str(8 - row), True, TEXT_COLOR)
                screen.blit(text, (BORDER_SIZE//2-10, y + SQUARE_SIZE//3))
            
            if row == 7:  # File letters
                font = pygame.font.SysFont('Arial', 20, True)
                text = font.render(chr(97 + col), True, TEXT_COLOR)
                screen.blit(text, (x + SQUARE_SIZE//3, HEIGHT + BORDER_SIZE*1.3))

# Enhanced piece drawing with smooth centering
def draw_pieces(screen, board, pieces):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            square = chess.square(col, 7 - row)
            piece = board.piece_at(square)
            if piece:
                color = 'w' if piece.color == chess.WHITE else 'b'
                piece_type = piece.symbol().lower()
                key = color + piece_type
                piece_img = pieces.get(key)
                if piece_img:
                    x = col * SQUARE_SIZE + BORDER_SIZE + 5  # Center offset
                    y = row * SQUARE_SIZE + BORDER_SIZE + 5  # Center offset
                    screen.blit(piece_img, (x, y))

# Enhanced square highlighting with animations
def highlight_square(screen, square):
    col = chess.square_file(square)
    row = 7 - chess.square_rank(square)
    x = col * SQUARE_SIZE + BORDER_SIZE
    y = row * SQUARE_SIZE + BORDER_SIZE
    
    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(highlight_surface, HIGHLIGHT, highlight_surface.get_rect())
    screen.blit(highlight_surface, (x, y))

# Enhanced move highlighting with elegant indicators
def highlight_legal_moves(screen, board, square):
    legal_moves = [move for move in board.legal_moves if move.from_square == square]
    for move in legal_moves:
        to_square = move.to_square
        col = chess.square_file(to_square)
        row = 7 - chess.square_rank(to_square)
        x = col * SQUARE_SIZE + BORDER_SIZE
        y = row * SQUARE_SIZE + BORDER_SIZE
        
        # Draw elegant move indicator
        center = (x + SQUARE_SIZE//2, y + SQUARE_SIZE//2)
        radius = SQUARE_SIZE//6
        
        highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        if board.piece_at(to_square):  # Capture move
            pygame.draw.circle(highlight_surface, MOVE_HIGHLIGHT, 
                            (SQUARE_SIZE//2, SQUARE_SIZE//2), SQUARE_SIZE//2, 3)
        else:  # Regular move
            pygame.draw.circle(highlight_surface, MOVE_HIGHLIGHT, 
                            (SQUARE_SIZE//2, SQUARE_SIZE//2), radius)
        screen.blit(highlight_surface, (x, y))

# Enhanced status display with animations
def display_status(screen, board):
    status = ""
    # Check for checkmate first
    if board.is_checkmate():
        # If White's turn and it's checkmate, then Black won (and vice versa)
        winner = "Black" if board.turn == chess.WHITE else "White"
        status = f"{winner} WINS BY CHECKMATE!"
    # Check for other game end conditions
    elif board.is_stalemate():
        status = "GAME OVER - Draw by stalemate"
    elif board.is_insufficient_material():
        status = "GAME OVER - Draw by insufficient material"
    elif board.is_fivefold_repetition():
        status = "GAME OVER - Draw by repetition"
    elif board.is_fifty_moves():
        status = "GAME OVER - Draw by fifty-move rule"
    elif board.is_check():
        checked = "White" if board.turn == chess.WHITE else "Black"
        status = f"{checked} is in check!"
    else:
        status = "White to move" if board.turn == chess.WHITE else "Black to move"

    # Create elegant status bar with enhanced visibility
    status_height = 60  # Increased height for better visibility
    status_y = HEIGHT + BORDER_SIZE*2 - status_height
    status_surface = pygame.Surface((WIDTH + 2*BORDER_SIZE, status_height), pygame.SRCALPHA)
    
    # Use different background colors for different states
    if "WINS" in status:
        bg_color = (0, 100, 0, 230)  # Green for victory
    elif "GAME OVER" in status:
        bg_color = (139, 69, 19, 230)  # Brown for draw
    elif "check" in status:
        bg_color = (139, 0, 0, 230)  # Red for check
    else:
        bg_color = (0, 0, 0, 180)  # Default dark background
    
    pygame.draw.rect(status_surface, bg_color, status_surface.get_rect())
    screen.blit(status_surface, (0, status_y))

    # Render status text with enhanced visibility
    font = pygame.font.SysFont('Arial', 32, True)  # Larger, bold font
    text = font.render(status, True, WHITE)
    text_rect = text.get_rect(center=(WIDTH//2 + BORDER_SIZE, status_y + status_height//2))
    screen.blit(text, text_rect)

# Add analysis display function
def display_analysis(screen, analysis, width, border_size):
    # Create a background surface for the analysis panel
    analysis_height = HEIGHT
    analysis_surface = pygame.Surface((ANALYSIS_WIDTH, analysis_height))
    analysis_surface.fill((40, 44, 52))  # Dark background color
    
    # Always blit the background surface first to clear previous text
    screen.blit(analysis_surface, (width + border_size, border_size))
    
    # Always show "Waiting for analysis..." at the top
    font = pygame.font.SysFont('Arial', 18, True)
    waiting_text = font.render("Waiting for analysis...", True, TEXT_COLOR)
    screen.blit(waiting_text, (width + border_size + 10, 20))
    
    if analysis:
        # Define colors
        LABEL_COLOR = (170, 170, 170)
        VALUE_COLOR = (200, 255, 200)
        WARNING_COLOR = (255, 150, 150)
        BEST_LINE_COLOR = (255, 200, 100)
        
        y_pos = 60  # Start below waiting text
        line_height = 22  # Reduced line height for compact display
        
        # Helper function to draw a line of text
        def draw_line(label, value, color=VALUE_COLOR):
            nonlocal y_pos
            label_surf = font.render(f"{label}:", True, LABEL_COLOR)
            value_surf = font.render(str(value), True, color)
            screen.blit(label_surf, (width + border_size + 10, y_pos))
            screen.blit(value_surf, (width + border_size + 140, y_pos))
            y_pos += line_height
        
        # Display move information
        draw_line("Move Number", analysis['MoveNumber'])
        draw_line("Move", analysis['Move'])
        draw_line("Current Player", analysis['CurrentPlayer'])
        draw_line("Phase", analysis['Phase'])
        
        y_pos += 5  # Add small spacing
        
        # Display evaluation information
        draw_line("Evaluation", analysis['Evaluation'])
        draw_line("Score Change", analysis['ScoreChange'])
        draw_line("Classification", analysis['Classification'])
        
        y_pos += 5  # Add small spacing
        
        # Display best line with word wrap
        label_surf = font.render("Best Line:", True, LABEL_COLOR)
        screen.blit(label_surf, (width + border_size + 10, y_pos))
        
        # Word wrap for best line
        best_line_words = analysis['BestLine'].split(", ")
        best_line = ""
        x_pos = width + border_size + 140
        for word in best_line_words:
            test_line = best_line + word + ", "
            test_surf = font.render(test_line, True, BEST_LINE_COLOR)
            if test_surf.get_width() > ANALYSIS_WIDTH - 150:
                if best_line:
                    line_surf = font.render(best_line, True, BEST_LINE_COLOR)
                    screen.blit(line_surf, (x_pos, y_pos))
                    y_pos += line_height
                best_line = word + ", "
            else:
                best_line = test_line
        if best_line:
            line_surf = font.render(best_line.rstrip(", "), True, BEST_LINE_COLOR)
            screen.blit(line_surf, (x_pos, y_pos))
        
        y_pos += line_height + 5  # Add spacing after best line
        
        # Display threat and move details
        draw_line("Mate Threat", analysis['MateThreat'])
        draw_line("Depth", analysis['Depth'])
        if analysis['Captured']:
            draw_line("Captured", analysis['Captured'])
        draw_line("Move Type", analysis['MoveType'].title())
        
        y_pos += 5  # Add small spacing
        
        # Display game state
        draw_line("Check", analysis['Check'])
        draw_line("Checkmate", analysis['Checkmate'])
        draw_line("Game Over", analysis['GameOver'])
        if analysis['GameOver'] == "Yes":
            draw_line("Game Over Reason", analysis['GameOverReason'], WARNING_COLOR)
            if analysis['Winner'] != "None":
                draw_line("Winner", analysis['Winner'], WARNING_COLOR)

def format_analysis_for_commentary(analysis):
    if not analysis:
        return ""
    
    # Format the analysis data in a way that matches the training data format
    formatted_input = f"MoveNumber: {analysis['MoveNumber']} | "
    formatted_input += f"Move: {analysis['Move']} | "
    formatted_input += f"CurrentPlayer: {analysis['CurrentPlayer']} | "
    formatted_input += f"Phase: {analysis['Phase']} | "
    formatted_input += f"Classification: {analysis['Classification']} | "
    formatted_input += f"MateThreat: {analysis['MateThreat']} | "
    formatted_input += f"Captured: {analysis['Captured']} | "
    formatted_input += f"MoveType: {analysis['MoveType']} | "
    formatted_input += f"Check: {analysis['Check']} | "
    formatted_input += f"Checkmate: {analysis['Checkmate']} | "
    formatted_input += f"GameOver: {analysis['GameOver']} | "
    formatted_input += f"GameOverReason: {analysis['GameOverReason']} | "
    formatted_input += f"Winner: {analysis['Winner']}"
    
    return formatted_input

# Import threading at the top of the file
import threading
import queue

# Speech thread with error handling
def speech_thread(engine, text_queue):
    while True:
        try:
            text = text_queue.get(timeout=0.1)  # Short timeout to check for new text
            if text is None:  # Sentinel value to stop the thread
                break
                
            # Clear the queue if it's getting too full to prevent lag
            while not text_queue.empty() and text_queue.qsize() > 2:
                text = text_queue.get()
                
            try:
                engine.say(text)
                engine.runAndWait()
            except RuntimeError as e:
                print(f"Speech engine error: {e}")
                continue  # Continue with next text if there's an error
            except Exception as e:
                print(f"Unexpected error in speech thread: {e}")
                continue
                
        except queue.Empty:
            continue  # No new text to process
        except Exception as e:
            print(f"Critical error in speech thread: {e}")
            break  # Stop the thread on critical errors
            
    try:
        engine.stop()  # Clean up the speech engine
    except:
        pass  # Ignore cleanup errors

# Add commentary generation thread
def commentary_thread(model, engine, queue_in, queue_out):
    while True:
        try:
            analysis_text = queue_in.get(timeout=0.1)
            if analysis_text is None:  # Stop signal
                break
                
            # Prepare input for casual, natural commentary like in training data
            entry = {
                "output": "Okay, here we go!",  # Placeholder to guide the style
                "input": analysis_text
            }
            input_text = format_input(entry)
            input_ids = tokenizer.encode(input_text)
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output_ids = generate(
                    model=model,
                    idx=input_tensor,
                    max_new_tokens=60,  # Longer to match training data style
                    context_size=1024,
                    temperature=1.2,  # More creative to match casual style
                    top_k=40  # Allow more variety in responses
                )
            
            commentary = tokenizer.decode(output_ids[0].tolist())
            # Clean up any potential markers or unwanted text
            if "### Response:" in commentary:
                commentary = commentary.split("### Response:")[-1].strip()
            if "### Instruction:" in commentary:
                commentary = commentary.split("### Instruction:")[0].strip()
            if "### Input:" in commentary:
                commentary = commentary.split("### Input:")[0].strip()
            if "###" in commentary:
                commentary = commentary.split("###")[0].strip()
            
            # Further cleanup of any remaining markers
            commentary = commentary.strip().replace("Instructions:", "").replace("Input:", "").replace("Response:", "")
            
            # Queue the commentary immediately as a single piece
            if commentary and not commentary.isspace():
                queue_out.put(commentary.strip() + '.')
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Commentary generation error: {e}")
            continue

# Update generate_commentary to use the speech queue
def generate_commentary(model, engine, analysis_text, text_queue):
    if not analysis_text:
        return
        
    # Match the casual style from training data
    entry = {
        "output": "Okay, here we go!",  # Placeholder to guide the style
        "input": analysis_text
    }
    input_text = format_input(entry)
    input_ids = tokenizer.encode(input_text)
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Use settings that match the training data style
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_tensor,
            max_new_tokens=60,  # Allow longer, more natural commentary
            context_size=1024,
            temperature=1.2,  # More creative to match casual style
            top_k=40  # Allow variety in responses
        )
    
    commentary = tokenizer.decode(output_ids[0].tolist())
    
    # Clean up everything
    if "### Response:" in commentary:
        commentary = commentary.split("### Response:")[-1].strip()
        
    if "### Instruction:" in commentary:
        commentary = commentary.split("### Instruction:")[0].strip()
        
    if "### Input:" in commentary:
        commentary = commentary.split("### Input:")[0].strip()
        
    if "###" in commentary:
        commentary = commentary.split("###")[0].strip()
    
    # Further cleanup of any remaining markers
    commentary = commentary.strip()
    commentary = commentary.replace("Instructions:", "")
    commentary = commentary.replace("Input:", "")
    commentary = commentary.replace("Response:", "")
    
    # Queue the commentary as a single piece
    if commentary and not commentary.isspace():
        text_queue.put(commentary.strip() + ".")

def main():
    board = chess.Board()
    analyzer = ChessAnalyzer()
    piece_images = load_piece_images()
    selected_square = None
    game_over = False
    current_analysis = None

    # Initialize commentary generation
    commentary_model = GPTModel(NEW_CONFIG)
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.pth')
    commentary_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    commentary_model.to(device)
    commentary_model.eval()

    # Initialize text-to-speech engine with faster settings
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 275)  # Increased speed for faster commentary
    tts_engine.setProperty('volume', 1.0)
    
    # Try to set a more natural voice
    voices = tts_engine.getProperty('voices')
    if voices:
        # Try to find a male voice for clearer commentary
        male_voices = [v for v in voices if 'david' in v.name.lower() or 'mark' in v.name.lower()]
        if male_voices:
            tts_engine.setProperty('voice', male_voices[0].id)

    # Initialize pygame screen
    screen = pygame.display.set_mode(
        (WIDTH + BORDER_SIZE * 2 + ANALYSIS_WIDTH, HEIGHT + BORDER_SIZE * 2)
    )
    pygame.display.set_caption("Chess Game with Commentary")
    
    # Set up queues and threads
    analysis_queue = queue.Queue()
    speech_queue = queue.Queue()
    
    # Start commentary thread
    commentary_thread_obj = threading.Thread(
        target=commentary_thread,
        args=(commentary_model, tts_engine, analysis_queue, speech_queue),
        daemon=True
    )
    commentary_thread_obj.start()
    
    # Start speech thread
    speech_thread_obj = threading.Thread(
        target=speech_thread,
        args=(tts_engine, speech_queue),
        daemon=True
    )
    speech_thread_obj.start()

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                # Stop threads gracefully
                analysis_queue.put(None)
                speech_queue.put(None)
                commentary_thread_obj.join(timeout=1.0)
                speech_thread_obj.join(timeout=1.0)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not game_over:
                    x, y = event.pos
                    file = (x - BORDER_SIZE) // SQUARE_SIZE
                    rank = (y - BORDER_SIZE) // SQUARE_SIZE

                    if 0 <= file < 8 and 0 <= rank < 8:
                        square = chess.square(file, 7 - rank)

                        if selected_square is None:
                            # First click - select piece
                            piece = board.piece_at(square)
                            if piece and piece.color == board.turn:
                                selected_square = square
                        else:
                            # Second click - try to move
                            move = chess.Move(selected_square, square)
                            legal_moves = list(board.legal_moves)
                            
                            # Handle promotion
                            promotion_moves = [m for m in legal_moves if m.from_square == selected_square 
                                            and m.to_square == square and m.promotion]
                            if promotion_moves:
                                move = promotion_moves[0]  # Default to queen promotion

                            # Try to make the move
                            if move in legal_moves or any(m.from_square == selected_square 
                                and m.to_square == square for m in legal_moves):
                                board.push(move)
                                
                                # Generate analysis and queue commentary asynchronously
                                current_analysis = analyzer.analyze_position(board, move)
                                analysis_text = format_analysis_for_commentary(current_analysis)
                                
                                # Clear old items from queue
                                while not analysis_queue.empty():
                                    try:
                                        analysis_queue.get_nowait()
                                    except queue.Empty:
                                        break
                                
                                # Queue new analysis
                                analysis_queue.put(analysis_text)
                                
                            selected_square = None

        # Draw everything
        clock.tick(FPS)
        screen.fill(BORDER_COLOR)
        draw_board(screen)
        draw_pieces(screen, board, piece_images)
        
        if selected_square is not None:
            highlight_square(screen, selected_square)
            highlight_legal_moves(screen, board, selected_square)
        
        display_status(screen, board)
        if current_analysis:
            display_analysis(screen, current_analysis, WIDTH, BORDER_SIZE)

        pygame.display.flip()

    # Clean up
    analyzer.close()
    pygame.quit()

if __name__ == "__main__":
    main()