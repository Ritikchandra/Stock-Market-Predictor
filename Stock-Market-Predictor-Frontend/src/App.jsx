import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [stockName, setStockName] = useState('');
  const [result, setResult] = useState(null);
  const [upload, setUpload] = useState(false)
  const analyzeStock = async () => {
    setUpload(true)
    const response = await axios.post('http://127.0.0.1:5000/analyze', {
      stock_name: stockName
    });
    setResult(response.data);
  };

  return (
    <div className="App">
      <div class='ripple-background'>
  <div class='circle xxlarge shade1'></div>
  <div class='circle xlarge shade2'></div>
  <div class='circle large shade3'></div>
  <div class='circle mediun shade4'></div>
  <div class='circle small shade5'></div>
</div>

      <header className="App-header">
        <h1>Stock Sentiment Analysis</h1>
        <div className="inputWrapper">
        <input
          type="text"
          placeholder="Enter Stock Name"
          value={stockName}
          onChange={(e) => setStockName(e.target.value)}
        />
        <button onClick={analyzeStock}>Analyze</button>
        </div>
        <div className='resultWrapper'>
          {!result && upload && (
            <>
            <span className='waitText'>Processing the input... Please Wait!</span>
            </>
          )}
        {result && (
          <>
            <h2 style={{color: result.prediction === 'down' ? 'darkred' : 'lightgreen', fontWeight: 'bold'}}>Prediction: The stock will go <span style={{color: result.prediction === 'down' ? 'darkred' : 'lightgreen', fontWeight: 'bold'}}>{result.prediction}</span>!</h2>
            <div className="articlesWrapper">
            <h3>Articles:</h3>
            <div className="articleWrappers">
              {result.urls.map((url, index) => (
                <div className="articleLinks" key={index}>
                  <a href={url} target="_blank" rel="noopener noreferrer">
                    {url}
                  </a>
                  </div>
              ))}
              </div>
              </div>
              <div className="scoreWrapper">
            <h3>Average Sentiment Scores:</h3>
            <div className="scoresWrapper">
              {Object.keys(result.avg_score).map((key) => (
                <div key={key} className='resultValues'>
                  {key}: {result.avg_score[key]}
                  </div>
              ))}
              </div>
              </div>
          </>
        )}
        </div>
      </header>
    </div>
  );
}

export default App;
