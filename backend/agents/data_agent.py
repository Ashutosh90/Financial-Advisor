"""
Data Agent - Fetches and processes market data
Responsibilities: Get FD rates, mutual fund NAVs, inflation data, RBI rates
"""
import yfinance as yf
import requests
from typing import Dict, Optional
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
from bs4 import BeautifulSoup
import re


class DataAgent:
    """Agent responsible for fetching financial market data"""
    
    def __init__(self):
        self.cache_duration = timedelta(hours=6)
        self._cache = {}

    def get_fd_rates(self) -> Dict[str, float]:
        """
        Get Fixed Deposit rates from HDFC Bank
        Scrapes real data from HDFC website with fallback to static rates
        Returns rates for various tenures from 1 month to 10 years
        Includes both regular and senior citizen rates
        """
        try:
            url = "https://www.hdfc.bank.in/fixed-deposit/fd-interest-rate"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            fd_rates = {}
            fd_rates_senior = {}
            
            # Find all tables on the page
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cols = row.find_all('td')
                    
                    if len(cols) >= 3:  # Need at least 3 columns: tenure, regular rate, senior rate
                        tenure_text = cols[0].get_text(strip=True).lower()
                        
                        try:
                            # Extract regular rate from second column
                            rate_text = cols[1].get_text(strip=True)
                            rate_match = re.search(r'(\d+\.?\d*)', rate_text)
                            if rate_match:
                                rate = float(rate_match.group(1))
                            else:
                                continue
                            
                            # Extract senior citizen rate from third column
                            senior_rate_text = cols[2].get_text(strip=True)
                            senior_rate_match = re.search(r'(\d+\.?\d*)', senior_rate_text)
                            if senior_rate_match:
                                senior_rate = float(senior_rate_match.group(1))
                            else:
                                senior_rate = rate  # Fallback to regular rate
                            
                            # Map tenures to standard keys
                            tenure_key = None
                            
                            # 1 month (30-45 days)
                            if '30' in tenure_text and '45' in tenure_text:
                                tenure_key = '1_month'
                            # 2 months (46-60 days)
                            elif '46' in tenure_text and '60' in tenure_text:
                                tenure_key = '2_months'
                            # 3 months (61-89 days or 90 days)
                            elif ('61' in tenure_text and '89' in tenure_text) or ('90 days' in tenure_text):
                                tenure_key = '3_months'
                            # 6 months
                            elif '6 months' in tenure_text and '9 months' not in tenure_text:
                                if 'to' not in tenure_text or '1 days' in tenure_text:
                                    tenure_key = '6_months'
                            # 9 months
                            elif '9 months' in tenure_text and '1 day' in tenure_text:
                                tenure_key = '9_months'
                            # 1 year
                            elif '1 year' in tenure_text and '15 months' not in tenure_text:
                                tenure_key = '1_year'
                            # 15 months
                            elif '15 months' in tenure_text and '18 months' in tenure_text:
                                tenure_key = '15_months'
                            # 18 months
                            elif '18 months' in tenure_text and '2 years' not in tenure_text:
                                tenure_key = '18_months'
                            # 2 years
                            elif '2 years' in tenure_text or '2 year' in tenure_text:
                                if '3 years' not in tenure_text and '3 year' not in tenure_text:
                                    tenure_key = '2_years'
                            # 3 years
                            elif '3 years' in tenure_text or '3 year' in tenure_text:
                                if '5 years' not in tenure_text and '5 year' not in tenure_text:
                                    tenure_key = '3_years'
                            # 5 years
                            elif '5 years' in tenure_text or '5 year' in tenure_text:
                                if '10 years' not in tenure_text and '10 year' not in tenure_text:
                                    tenure_key = '5_years'
                            # 10 years
                            elif '10 years' in tenure_text or '10 year' in tenure_text:
                                tenure_key = '10_years'
                            
                            if tenure_key:
                                fd_rates[tenure_key] = rate
                                fd_rates_senior[tenure_key] = senior_rate
                            
                        except (ValueError, AttributeError, IndexError) as e:
                            logger.debug(f"Error parsing row: {e}")
                            continue
            
            # Return rates if we got any
            if fd_rates:
                result = {
                    "regular": fd_rates,
                    "senior_citizen": fd_rates_senior
                }
                logger.info(f"Fetched FD rates from HDFC Bank: Regular - {fd_rates}, Senior - {fd_rates_senior}")
                return result
            
            # If scraping failed or didn't get enough data, use fallback
            raise Exception("Could not parse FD rates from HDFC website")
            
        except Exception as e:
            logger.warning(f"Error fetching FD rates from HDFC website: {e}. Using fallback rates.")
            # Fallback to realistic static rates (as of Dec 2024)
            fd_rates = {
                "1_month": 3.25,
                "2_months": 4.25,
                "3_months": 4.25,
                "6_months": 5.50,
                "9_months": 5.75,
                "1_year": 6.25,
                "15_months": 6.35,
                "18_months": 6.50,
                "2_years": 6.75,
                "3_years": 7.00,
                "5_years": 7.00,
                "10_years": 7.00
            }
            fd_rates_senior = {
                "1_month": 3.75,
                "2_months": 4.75,
                "3_months": 4.75,
                "6_months": 6.00,
                "9_months": 6.25,
                "1_year": 6.75,
                "15_months": 6.85,
                "18_months": 7.00,
                "2_years": 7.25,
                "3_years": 7.50,
                "5_years": 7.50,
                "10_years": 7.50
            }
            result = {
                "regular": fd_rates,
                "senior_citizen": fd_rates_senior
            }
            logger.info(f"Using fallback FD rates: {result}")
            return result
    
    def get_repo_rate(self) -> float:
        """
        Get current RBI Repo Rate from Reserve Bank of India website
        Scrapes real data from RBI website with fallback to static rate
        """
        try:
            # Try to fetch from RBI website
            url = "https://www.rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx?id=12647"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for repo rate in the page content
                # RBI typically displays current policy rates in tables or specific divs
                text_content = soup.get_text().lower()
                
                # Try to find repo rate mentions
                if 'repo rate' in text_content:
                    # Find all text containing numbers that might be the rate
                    for element in soup.find_all(['td', 'p', 'div', 'span']):
                        text = element.get_text(strip=True).lower()
                        if 'repo rate' in text:
                            # Try to extract the rate from nearby elements
                            parent = element.find_parent(['tr', 'table', 'div'])
                            if parent:
                                rate_text = parent.get_text()
                                # Look for percentage patterns like "6.50%" or "6.5 percent"
                                matches = re.findall(r'(\d+\.?\d*)\s*(?:%|per\s*cent|percent)', rate_text)
                                if matches:
                                    for match in matches:
                                        rate = float(match)
                                        if 4.0 <= rate <= 10.0:  # Reasonable repo rate range
                                            logger.info(f"Fetched Repo Rate from RBI: {rate}%")
                                            return rate
                
                # Alternative: Try RBI's current rates page
                alt_url = "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=57068"
                response = requests.get(alt_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    text = soup.get_text()
                    matches = re.findall(r'repo\s+rate.*?(\d+\.?\d*)\s*(?:%|per\s*cent)', text.lower())
                    if matches:
                        rate = float(matches[0])
                        if 4.0 <= rate <= 10.0:
                            logger.info(f"Fetched Repo Rate from RBI: {rate}%")
                            return rate
            
            # If scraping failed, use fallback
            raise Exception("Could not parse Repo Rate from RBI website")
            
        except Exception as e:
            logger.warning(f"Error fetching Repo Rate from RBI website: {e}. Using fallback rate.")
            # Fallback to current repo rate (as of Dec 2024)
            repo_rate = 6.5
            logger.info(f"Using fallback Repo Rate: {repo_rate}%")
            return repo_rate
    
    def get_inflation_rate(self) -> float:
        """
        Get current inflation rate (CPI)
        In production, fetch from RBI/government sources
        """
        try:
            # Simulated inflation rate
            inflation_rate = 5.4
            logger.info(f"Current Inflation Rate: {inflation_rate}%")
            return inflation_rate
        except Exception as e:
            logger.error(f"Error fetching inflation rate: {e}")
            return 5.0
    
    def get_mutual_fund_returns(self, category: str = "debt") -> Dict[str, float]:
        """
        Get average mutual fund returns by category
        Categories: debt, equity, hybrid, liquid
        """
        try:
            # Simulated historical returns (1-year average)
            returns_data = {
                "debt": {
                    "short_term": 6.8,
                    "long_term": 7.5,
                    "liquid": 5.5
                },
                "equity": {
                    "large_cap": 12.5,
                    "mid_cap": 15.0,
                    "small_cap": 18.0,
                    "index_fund": 11.0
                },
                "hybrid": {
                    "aggressive": 10.5,
                    "conservative": 8.0,
                    "balanced": 9.5
                },
                "liquid": {
                    "liquid_fund": 5.5
                }
            }
            
            result = returns_data.get(category, returns_data["debt"])
            logger.info(f"Fetched MF returns for {category}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error fetching mutual fund returns: {e}")
            return {"short_term": 6.5}
    
    def get_index_data(self, index: str = "^NSEI") -> Optional[Dict]:
        """
        Get stock index data using yfinance
        Default: NIFTY 50 (^NSEI)
        """
        try:
            ticker = yf.Ticker(index)
            info = ticker.info
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            month_start_price = hist['Close'].iloc[0]
            monthly_return = ((current_price - month_start_price) / month_start_price) * 100
            
            data = {
                "index": index,
                "current_price": float(current_price),
                "monthly_return": float(monthly_return),
                "52_week_high": float(info.get("fiftyTwoWeekHigh", 0)),
                "52_week_low": float(info.get("fiftyTwoWeekLow", 0))
            }
            
            logger.info(f"Fetched index data for {index}")
            return data
        except Exception as e:
            logger.error(f"Error fetching index data: {e}")
            return None
    
    def get_gold_rates(self) -> Dict[str, float]:
        """
        Get current gold prices from Groww website
        Returns 24K gold rate for 10 grams in INR
        """
        try:
            # Fetch from Groww
            url = "https://groww.in/gold-rates/gold-rate-today-in-pune"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            import re
            
            # Get page text
            page_text = soup.get_text()
            
            # Pattern to match: 24K followed by Gold / 10gm ₹XXXXX.XX
            match = re.search(r'24K.*?10gm.*?₹\s*([\d,]+(?:\.\d{2})?)', page_text, re.DOTALL)
            
            if match:
                price_str = match.group(1).replace(',', '')
                price_per_10_gram_24k = float(price_str)
                
                # Round to nearest integer for cleaner display (gold prices don't need decimal precision)
                price_per_10_gram_24k = round(price_per_10_gram_24k)
                
                logger.info(f"Scraped gold rate from Groww: ₹{price_per_10_gram_24k:,.0f}/10g (24K)")
                
                return {
                    "price_per_10_gram_24k": price_per_10_gram_24k
                }
            
            raise Exception("Could not find 24K gold rate in Groww page")
            
        except Exception as e:
            logger.warning(f"Error fetching gold rate from Groww: {e}. Using fallback.")
            # Fallback to realistic static rate (as of Dec 2025)
            return {
                "price_per_10_gram_24k": 131480.0
            }
    
    def get_all_market_data(self) -> Dict:
        """
        Aggregate all market data
        """
        market_data = {
            "fd_rates": self.get_fd_rates(),
            "repo_rate": self.get_repo_rate(),
            "inflation_rate": self.get_inflation_rate(),
            "mf_returns": {
                "debt": self.get_mutual_fund_returns("debt"),
                "equity": self.get_mutual_fund_returns("equity"),
                "hybrid": self.get_mutual_fund_returns("hybrid")
            },
            "gold_rates": self.get_gold_rates(),
            "nifty_data": self.get_index_data("^NSEI"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("Aggregated all market data")
        return market_data