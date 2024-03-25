import React, { useState, useEffect, useRef, Fragment } from 'react';
import { useLocation } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import AnalysisMoves from './AnalysisMoves';
import Bar from '../components/Bar';
import './Analyzed.css';

const Analyzed = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const [index, setIndex] = useState(0);
  // const moveIndex = useRef(0);
  const [moves, setMoves] = useState([]);
  const location = useLocation();
  const pgn = location.state.pgn;
  const moveIndexToFeedback = useRef({});
  let arrows = [];
  let feedback = "";

  const getPgnMoves = (pgn) => {
    const newChess = new Chess();
    newChess.loadPgn(pgn);
    console.log(newChess.history());
    return newChess.history();
  };

  // TODO: tony can you create these three functions: 
  // nextMove(current) -- basically what you do with arrow right rn but takes in current and then updates chess board with newest move
  // updateBoard(move) -- takes in the index of a move and updates the board to that state in the game 
  // prevMove() -- basically what you do with arrow left rn and undoes a move
  const makeMove = () => {
    chess.current.move(moves[index]);
    setFen(chess.current.fen());
    setIndex(index + 1);

    // if not cached, request for feedback
    if (!moveIndexToFeedback.current[index]) {
      const model_input = [];
      const history = chess.history({ verbose: true }).slice(0, index);
      const moves_made = moves.slice(0, index); // gets all moves so far
      for (let i = 0; i < index; i++) {
        model_input.append((history[i].from, moves_made.slice(Math.max(0, i - 16), i), history[i].color));
      }

      // get feedback from model
      // const feedback = model(model_input);
    } else {
      // display cached feedback
      feedback = moveIndexToFeedback.current[index.current];
    }
    feedback = "Feedback";

    // store feedback in hashmap
    moveIndexToFeedback.current[index] = feedback;
  }

  const undoMove = () => {
    chess.current.undo();
    setFen(chess.current.fen());
    setIndex(index - 1);

    // get cached feedback
    feedback = moveIndexToFeedback.current[index];
    
    // display feedback
  }

  useEffect(() => {
    const handleKeydown = (event) => {
      if (event.key === 'ArrowRight' && index < moves.length) {
        makeMove();
      } else if (event.key === 'ArrowLeft' && index > 0) {
        undoMove();
      }
    };

    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [moves, index])

  useEffect(() => {
    setMoves(getPgnMoves(pgn)); // format: ['e4', 'e5', ..., 'Nf3', 'Nc6']
  }, [pgn]);

  return (
    <div className="analysis-page-layout">
      <div className="analysis-header">
        <h2>Analysis</h2>
      </div>
      <div className="analysis-page-container">
        {/* <Progress.Line showInfo={false} strokeColor={"white"} vertical={true}/> */}
        <Bar color={"black"} value={50} label={"+0.0"}/>
        <Bar color={"blue"} value={80} label={"hard"}/>
        <div className="chessboard-container">
          <Chessboard position={fen} areArrowsAllowed={false} arePiecesDraggable={false} boardWidth={560} customArrows={[arrows]}/>
        </div>
        <div className="analysis-right-container">
          <AnalysisMoves moves={moves} index={index}/>
          <div className="analysis-button-container">
            <div className="analysis-back" onClick={undoMove}></div>
            <div className="analysis-move" onClick={makeMove}></div>
          </div>
        </div>
      </div>
    </div>
  );
};  
export default Analyzed;