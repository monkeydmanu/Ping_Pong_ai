# Quick Start: Mouse Control for Player 1

## Latest Update
✅ Mouse control system is now **fully parameterized** across all game modes!

---

## How to Enable Mouse Control

### Option 1: Modify train.py (Easiest)

Edit the main block at the bottom of `train.py`:

```python
if __name__ == "__main__":
    # ... existing code ...
    
    # Human vs Human with MOUSE CONTROL
    play_human_vs_human(mouse_control_p1=True)
    
    # OR Human vs AI with MOUSE CONTROL
    # play_ai_vs_human(model_path='models/ppo', mouse_control_p1=True)
```

### Option 2: Direct Python API

```python
from train import play_human_vs_human, play_ai_vs_human

# Human vs Human - Mouse Control
play_human_vs_human(mouse_control_p1=True)

# Human vs AI - Mouse Control
play_ai_vs_human(model_path='models/ppo', mouse_control_p1=True)

# Human vs Human - Keyboard Control (default)
play_human_vs_human()  # mouse_control_p1 defaults to False
```

### Option 3: Direct Game Class

```python
from engine.game import Game

# Create game with mouse control enabled
game = Game(player1_type="human", player2_type="human", mouse_control_p1=True)
game.run()
```

---

## Control Schemes

### 🖱️ Mouse Control Mode (`mouse_control_p1=True`)

**Player 1 (Left Paddle):**
- **Position**: Move mouse cursor
- **Left Click**: Rotate paddle left
- **Right Click**: Rotate paddle right

**Player 2 (Right Paddle):** (Unchanged)
- **O/L**: Up/Down
- **K/M**: Left/Right  
- **I/P**: Rotate left/right

### ⌨️ Keyboard Control Mode (`mouse_control_p1=False`, Default)

**Player 1 (Left Paddle):**
- **Z/S**: Up/Down
- **Q/D**: Left/Right
- **A/E**: Rotate left/right

**Player 2 (Right Paddle):**
- **O/L**: Up/Down
- **K/M**: Left/Right
- **I/P**: Rotate left/right

---

## Key Features

✅ **Real-time Paddle Tracking**: Paddle follows mouse smoothly  
✅ **Velocity Smoothing**: 10-frame averaging prevents jitter  
✅ **Boundary Constraints**: Paddle stays within table  
✅ **Click-based Rotation**: Left/Right clicks for spin  
✅ **Backward Compatible**: Keyboard mode still available  
✅ **No Physics Breaking**: Velocity correctly calculated  

---

## Testing

Verify everything works:

```bash
python test_mouse_control.py
```

Expected output:
```
✅ ALL TESTS PASSED!

Summary:
  1. PingPongEnv accepts and stores player1_mouse_control parameter
  2. Game passes player1_mouse_control to PingPongEnv correctly
  3. _get_player1_input() uses conditional logic based on mouse_control_p1
  4. _apply_action() skips movement for agent in mouse mode
```

---

## Documentation

- **Implementation Details**: See [MOUSE_CONTROL_IMPLEMENTATION.md](MOUSE_CONTROL_IMPLEMENTATION.md)
- **Complete Summary**: See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Code Tests**: See [test_mouse_control.py](test_mouse_control.py)

---

## Files Modified

1. ✅ [train.py](train.py) - Added `mouse_control_p1` parameter
2. ✅ [engine/game.py](engine/game.py) - Conditional mouse/keyboard input logic
3. ✅ [ai/environment.py](ai/environment.py) - Conditional action application
4. ✅ New: [test_mouse_control.py](test_mouse_control.py) - Test suite
5. ✅ New: [MOUSE_CONTROL_IMPLEMENTATION.md](MOUSE_CONTROL_IMPLEMENTATION.md) - Detailed docs
6. ✅ New: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical summary

---

## Troubleshooting

**Issue**: Paddle not following mouse  
**Solution**: Make sure `mouse_control_p1=True` is passed and you have a human player 1

**Issue**: Rotation not working  
**Solution**: Try left/right mouse clicks (or check if right-click is captured by OS)

**Issue**: Paddle goes outside bounds  
**Solution**: This shouldn't happen - boundaries are enforced. Report if it does!

**Issue**: Jerky paddle movement  
**Solution**: Velocity smoothing is automatic (10-frame window). System is working as designed.

---

## Examples

### Play human vs human with mouse
```python
from train import play_human_vs_human
play_human_vs_human(mouse_control_p1=True)
```

### Play against AI with mouse
```python
from train import play_ai_vs_human
play_ai_vs_human(model_path='models/ppo', mouse_control_p1=True)
```

### Programmatic game creation
```python
from engine.game import Game
game = Game(player1_type="human", player2_type="ai", mouse_control_p1=True)
game.run()
```

---

## Implementation Status

| Feature | Status |
|---------|--------|
| Parameter in train.py | ✅ Complete |
| Parameter in Game class | ✅ Complete |
| Parameter in Environment | ✅ Complete |
| Mouse input handling | ✅ Complete |
| Keyboard fallback | ✅ Complete |
| Velocity calculation | ✅ Complete |
| Velocity smoothing | ✅ Complete |
| Rotation control | ✅ Complete |
| Boundary constraints | ✅ Complete |
| Tests | ✅ Complete |
| Documentation | ✅ Complete |

---

**Ready to play! 🎮**
