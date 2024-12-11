import React, { useState } from 'react'
import styled from 'styled-components'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const ResultsContainer = styled.div`
  margin-top: 2rem;
  padding: 1rem;
  background-color: #f7fafc;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
`

const ResultItem = styled.div`
  margin: 0.5rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
`

const MainResult = styled.div`
  padding: 1rem;
  background-color: white;
  border-radius: 8px;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
`

const Classification = styled.strong`
  color: #2b6cb0;
  font-size: 1.125rem;
`

const Confidence = styled.strong`
  color: #2f855a;
  font-size: 1.125rem;
`

const DetailButton = styled.button`
  margin-bottom: 1rem;
  padding: 0.5rem 1rem;
  background-color: #4299e1;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #3182ce;
  }
`

const DetailsSection = styled.div`
  margin-top: 1rem;
`

const ChartContainer = styled.div`
  height: 300px;
  width: 100%;
  margin: 1rem 0;
`

const DetailMessage = styled.p`
  color: #4a5568;
  font-style: italic;
  padding: 1rem;
  background-color: white;
  border-radius: 4px;
  margin-top: 1rem;
`

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
  color: #2d3748;
`

const SubTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  margin-bottom: 0.75rem;
  color: #4a5568;
`

const Results = ({ data }) => {
  const [showDetails, setShowDetails] = useState(false)

  // Prepare data for bar chart
  const chartData = data.details?.class_probabilities 
    ? Object.entries(data.details.class_probabilities).map(([label, value]) => ({
        name: label,
        probability: (value * 100).toFixed(2)
      }))
    : []

  return (
    <ResultsContainer>
      <Title>Analysis Results</Title>
      
      <MainResult>
        <ResultItem>
          <span>Classification:</span>
          <Classification>{data.classification}</Classification>
        </ResultItem>
        <ResultItem>
          <span>Confidence:</span>
          <Confidence>{(data.confidence * 100).toFixed(2)}%</Confidence>
        </ResultItem>
      </MainResult>

      {data.details?.class_probabilities && (
        <DetailsSection>
          <DetailButton onClick={() => setShowDetails(!showDetails)}>
            {showDetails ? 'Hide' : 'Show'} Detailed Analysis
          </DetailButton>
          
          {showDetails && (
            <>
              <SubTitle>Class Probabilities</SubTitle>
              <ChartContainer>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <XAxis dataKey="name" />
                    <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="probability" fill="#4299e1" />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
              
              {data.details.message && (
                <DetailMessage>{data.details.message}</DetailMessage>
              )}
            </>
          )}
        </DetailsSection>
      )}
    </ResultsContainer>
  )
}

export default Results