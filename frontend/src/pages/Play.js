import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import './Play.css';

const Play = () => {
  const navigate = useNavigate();
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const [color, setColor] = useState(null);
  const [turn, setTurn] = useState('white');
  const [isCheckmate, setIsCheckmate] = useState(false);
  const [isDraw, setIsDraw] = useState(false);
  const [elo, setElo] = useState(null);
  const [squareStyles, setSquareStyles] = useState([]);

  const playRandomMove = () => {
    const moves = chess.current.moves();
    const move = moves[Math.floor(Math.random() * moves.length)];
    chess.current.move(move);
    setFen(chess.current.fen());
    if (chess.current.isCheckmate()) {
      console.log('checkmate');
    } else if (chess.current.isDraw()) {
      console.log('draw');
    }
  };

  const handleColorChange = (newColor) => {
    setColor(newColor);
  };

  const handleEloChange = (newElo) => {
    setElo(newElo);
    if (color === 'black') {
      const opening_moves = ['e4', 'd4', 'Nf3']
      // play one out of the three
      const move = opening_moves[Math.floor(Math.random() * opening_moves.length)];
      setTimeout(() => {
        chess.current.move(move);
        setFen(chess.current.fen());
      }, 500);
    }
  }
  
  const getEngineMove = async () => {
    const no_moves = chess.current.history().length;
    try {
      const response = await fetch("http://localhost:8000/get-human-move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          fen: chess.current.fen(), 
          last_16_moves: chess.current.history().slice(Math.max(0, no_moves - 16), no_moves),
          is_white_move: chess.current.history().length % 2 !== 1, // if odd number of moves, then return false (black) since we want model to give black move
          elo: elo
        })
      });
      const data = await response.json();
      console.log(data);
      const moves = Object.keys(data);
      const probabilities = Object.values(data);
      const sumProb = probabilities.reduce((a, b) => a + b, 0);
      for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sumProb;
      }
      console.log(moves);
      console.log(probabilities);
      const threshold = Math.random();
      let runningProb = probabilities[0];
      let selectedMove = null;
      for (let i = 0; i < moves.length; i++) {
        if (runningProb > threshold) {
          selectedMove = moves[i];
          break;
        }
        runningProb += probabilities[i + 1];
      }

      if (selectedMove) {
        chess.current.move(selectedMove);
        setFen(chess.current.fen());
        if (chess.current.isCheckmate()) {
          console.log('checkmate');
          setIsCheckmate(true);
        } else if (chess.current.isDraw()) {
          console.log('draw');
          setIsDraw(true);
        }
      } else {
        console.log('error: no move selected');
      }
    } catch (error) {
      playRandomMove();
      console.log(error);
    }
  }

  const squareStyling = ({ pieceSquare, history }) => {
    const sourceSquare = history.length && history[history.length - 1].from;
    const targetSquare = history.length && history[history.length - 1].to;
    return {
      [pieceSquare]: { backgroundColor: "rgba(255, 255, 0, 0.4)" },
      ...(history.length && {
        [sourceSquare]: {
          backgroundColor: "rgba(255, 255, 0, 0.4)"
        }
      }),
      ...(history.length && {
        [targetSquare]: {
          backgroundColor: "rgba(255, 255, 0, 0.4)"
        }
      })
    };
  };
  

  // Function to handle piece click
  const onClick = (sourceSquare) => {
    setSquareStyles(squareStyling({ pieceSquare: sourceSquare, history: chess.current.history({ verbose: true }) }));
    // if (chess.current.get(sourceSquare) === null) return;
    // let legalMoves = chess.current.moves({ square: sourceSquare, verbose: true });
    // let squaresToHighlight = legalMoves.map(move => move.to);
    // let newSquareStyles = {};
    // squaresToHighlight.forEach(square => {
    //   newSquareStyles[square] = {
    //     backgroundColor: 'rgba(255, 255, 0, 0.4)',
    //     background: 'radial-gradient(circle, #fffc00 36%, transparent 40%)',
    //     borderRadius: '50%'
    //   };
    // });
    // console.log(newSquareStyles);
    // setSquareStyles(newSquareStyles);
  }
  
  // Function to handle piece drop
  const onDrop = (sourceSquare, targetSquare, piece) => {
    console.log(chess.current.ascii())
    try {
      let move = chess.current.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q'
      });
      if (move === null) return;
      setFen(chess.current.fen());
      if (chess.current.isCheckmate()) {
        console.log('checkmate');
        setIsCheckmate(true);
        return false;
      } else if (chess.current.isDraw()) {
        console.log('draw');
        setIsDraw(true);
        return false;
      }
      setTimeout(() => {
        getEngineMove();
      }, 500);
      if (chess.current.isCheckmate()) {
        console.log('checkmate');
        <input type="text" value="Game is over" readOnly />
      } else if (chess.current.isDraw()) {
        console.log('draw');
        <input type="text" value="Game is over" readOnly />
      }
    } catch (error) {
      console.log(error);
      return null;
    }
  };

  return (
    <div className="play-page-container">
      {!color && (<div className="buttons">
        <div>
          <button className="white-button" onClick={() => handleColorChange('white')}>play as white</button>
        </div>
        <div>
          <button className="black-button" onClick={() => handleColorChange('black')}>play as black</button>
        </div>
      </div>)}

      {color && !elo && (<div className="level-buttons">
        <div>
        <button className="level-500-button" onClick={() => handleEloChange(1500)}>play vs. 1100</button>
      </div>
      <div>
        <button className="level-1000-button" onClick={() => handleEloChange(1500)}>play vs. 1500</button>
      </div>
      <div>
        <button className="level-1500-button" onClick={() => handleEloChange(2100)}>play vs. 2100</button>
      </div>
    </div>)}
      {color && elo && (
        <div className="game-container">
          <div className="chessboard-container">
            <Chessboard
              position={fen}
              onPieceDrop={onDrop}
              onSquareClick={onClick}
              boardOrientation={color}
              boardWidth={560}
              arePiecesDraggable={true}
              squareStyles={squareStyles}
              />
          </div>
          <div className="game-buttons">
            {isCheckmate && <input type="text" value="Checkmate!" readOnly className="checkmate-box" />}
            {isDraw && <input type="text" value="Stalemate!" readOnly className="stalemate-box" />}
            <div><button className="reset-button" onClick={() => {
              chess.current.reset();
              setFen(chess.current.fen());
              setIsCheckmate(false);
              setIsDraw(false);
              // if (color === 'black') {
              //   handleColorChange('black');
              // }
              window.location.reload();
            }}>reset</button></div>
            <div><button className="undo-button" onClick={() => {
                  setIsCheckmate(false);
                  setIsDraw(false);
                  setTimeout(() => {
                    chess.current.undo();
                    setFen(chess.current.fen());
                  }, 200); // buffer time looks appropriate
                  setTimeout(() => {
                    chess.current.undo();
                    setFen(chess.current.fen());
                  }, 600); // undo twice to undo model move as well
                }}>undo</button></div>
            <div><button className="analyze-button" onClick={() => {
              navigate('/analyzed', { state: { pgn: chess.current.pgn() } });
            }}>analyze</button></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Play;