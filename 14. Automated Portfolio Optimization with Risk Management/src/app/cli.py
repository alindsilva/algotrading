"""
Command Line Interface for the IBKR trading application.
Provides commands for running the app, generating reports, and managing the system.
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .main import IBKRApp
from ..core.config import IBConfig, load_config
from src.core.logging import setup_logging

logger = logging.getLogger(__name__)


class IBKRCLIError(Exception):
    """CLI-specific errors"""
    pass


async def run_app_command(args) -> int:
    """Run the main IBKR application"""
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    logger.info("Starting IBKR trading application from CLI")
    
    try:
        # Initialize the application
        app = IBKRApp(config_path=args.config)
        
        # Start the application
        if not await app.start():
            logger.error("Failed to start application")
            return 1
        
        # Run until signal
        await app.run_until_signal()
        
        # Graceful shutdown
        await app.stop()
        
        logger.info("IBKR application terminated successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


async def status_command(args) -> int:
    """Get application status"""
    setup_logging(level='WARNING')  # Minimal logging for status
    
    try:
        # For now, just check if we can load config and connect
        config = load_config(args.config) if args.config else IBConfig()
        
        app = IBKRApp(config_path=args.config)
        status_info = {
            'config_loaded': True,
            'config_path': args.config,
            'host': config.host,
            'port': config.port,
            'database_path': str(config.database.path),
            'timestamp': datetime.now().isoformat()
        }
        
        if args.json:
            print(json.dumps(status_info, indent=2))
        else:
            print("IBKR Application Status:")
            print(f"  Config loaded: {status_info['config_loaded']}")
            print(f"  Config path: {status_info['config_path'] or 'Default'}")
            print(f"  IBKR Host: {status_info['host']}:{status_info['port']}")
            print(f"  Database: {status_info['database_path']}")
            print(f"  Timestamp: {status_info['timestamp']}")
        
        return 0
        
    except Exception as e:
        error_info = {'error': str(e), 'timestamp': datetime.now().isoformat()}
        
        if args.json:
            print(json.dumps(error_info, indent=2))
        else:
            print(f"Error checking status: {e}")
        
        return 1


async def portfolio_command(args) -> int:
    """Get portfolio summary"""
    setup_logging(level='WARNING')  # Minimal logging for portfolio command
    
    try:
        app = IBKRApp(config_path=args.config)
        
        # Start the application briefly to get portfolio data
        if not await app.start():
            raise IBKRCLIError("Failed to start application for portfolio analysis")
        
        try:
            # Get portfolio summary
            portfolio_data = await app.get_portfolio_summary()
            
            if args.json:
                # Convert datetime objects to strings for JSON serialization
                portfolio_json = json.loads(json.dumps(portfolio_data, default=str))
                print(json.dumps(portfolio_json, indent=2))
            else:
                print_portfolio_summary(portfolio_data)
            
            return 0
            
        finally:
            await app.stop()
    
    except Exception as e:
        error_info = {'error': str(e), 'timestamp': datetime.now().isoformat()}
        
        if args.json:
            print(json.dumps(error_info, indent=2))
        else:
            print(f"Error getting portfolio data: {e}")
        
        return 1


async def risk_report_command(args) -> int:
    """Generate risk report"""
    setup_logging(level='WARNING')  # Minimal logging for risk report
    
    try:
        app = IBKRApp(config_path=args.config)
        
        # Start the application briefly to get risk data
        if not await app.start():
            raise IBKRCLIError("Failed to start application for risk analysis")
        
        try:
            # Get risk report
            risk_data = await app.get_risk_report(days=args.days)
            
            if args.json:
                # Convert datetime and pandas objects to strings for JSON serialization
                risk_json = json.loads(json.dumps(risk_data, default=str))
                print(json.dumps(risk_json, indent=2))
            else:
                print_risk_report(risk_data)
            
            return 0
            
        finally:
            await app.stop()
    
    except Exception as e:
        error_info = {'error': str(e), 'timestamp': datetime.now().isoformat()}
        
        if args.json:
            print(json.dumps(error_info, indent=2))
        else:
            print(f"Error generating risk report: {e}")
        
        return 1


def print_portfolio_summary(portfolio_data: Dict[str, Any]):
    """Print formatted portfolio summary"""
    print("=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {portfolio_data['timestamp']}")
    
    # Connection status
    conn = portfolio_data.get('connection_status', {})
    print(f"\nConnection Status: {'✓ Connected' if conn.get('connected') else '✗ Disconnected'}")
    if conn.get('connected'):
        print(f"  Uptime: {conn.get('uptime_seconds', 0):.0f} seconds")
    
    # Account summary
    account = portfolio_data.get('account_summary', {})
    if account:
        print(f"\nAccount Summary:")
        if account.get('net_liquidation'):
            print(f"  Net Liquidation: ${account['net_liquidation']:,.2f}")
        if account.get('total_cash'):
            print(f"  Total Cash: ${account['total_cash']:,.2f}")
        if account.get('buying_power'):
            print(f"  Buying Power: ${account['buying_power']:,.2f}")
    
    # Positions
    positions = portfolio_data.get('positions', {})
    print(f"\nPositions Summary:")
    print(f"  Total Positions: {positions.get('total_positions', 0)}")
    print(f"  Long Exposure: ${positions.get('long_exposure', 0):,.2f}")
    print(f"  Short Exposure: ${positions.get('short_exposure', 0):,.2f}")
    print(f"  Net Exposure: ${positions.get('net_exposure', 0):,.2f}")
    
    # Largest positions
    largest = positions.get('largest_positions', [])
    if largest:
        print(f"\nLargest Positions:")
        for i, pos in enumerate(largest[:5], 1):
            print(f"  {i}. {pos['symbol']}: ${pos['value']:,.2f} ({pos['pct_of_portfolio']:.1%})")
    
    # Portfolio metrics
    metrics = portfolio_data.get('portfolio_metrics')
    if metrics and metrics.get('total_return') is not None:
        print(f"\nPortfolio Metrics:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  VaR (95%): {metrics['var_95']:.2%}")


def print_risk_report(risk_data: Dict[str, Any]):
    """Print formatted risk report"""
    print("=" * 60)
    print("RISK ANALYSIS REPORT")
    print("=" * 60)
    print(f"Timestamp: {risk_data['timestamp']}")
    print(f"Analysis Period: {risk_data['analysis_period_days']} days")
    
    # Check for errors
    if 'error' in risk_data:
        print(f"\nError: {risk_data['error']}")
        return
    
    # Portfolio metrics
    metrics = risk_data.get('portfolio_metrics', {})
    if metrics:
        print(f"\nPortfolio Risk Metrics:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"  Omega Ratio: {metrics['omega_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  VaR (95%): {metrics['var_95']:.2%}")
        print(f"  CVaR (95%): {metrics['cvar_95']:.2%}")
    
    # Rolling metrics
    rolling = risk_data.get('rolling_metrics', {})
    if rolling:
        print(f"\nRolling Metrics (Recent):")
        if rolling.get('latest_volatility'):
            print(f"  Latest Volatility: {rolling['latest_volatility']:.2%}")
        if rolling.get('latest_sharpe'):
            print(f"  Latest Sharpe Ratio: {rolling['latest_sharpe']:.3f}")
        if rolling.get('volatility_trend'):
            print(f"  Volatility Trend: {rolling['volatility_trend']}")
    
    # Risk alerts
    alerts = risk_data.get('risk_alerts', [])
    if alerts:
        print(f"\nRisk Alerts:")
        for alert in alerts:
            severity_symbol = "⚠️" if alert['severity'] == 'WARNING' else "ℹ️"
            print(f"  {severity_symbol} {alert['type']}: {alert['message']}")
    else:
        print(f"\nRisk Alerts: ✓ No alerts")
    
    # Stress test results
    stress = risk_data.get('stress_test_results', {})
    if stress:
        print(f"\nStress Test Results:")
        for scenario, results in stress.items():
            print(f"  {scenario.replace('_', ' ').title()}:")
            print(f"    Sharpe Ratio Impact: {results['sharpe_ratio_change']:+.3f}")
            print(f"    Max Drawdown Impact: {results['max_drawdown_change']:+.2%}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser"""
    parser = argparse.ArgumentParser(
        prog='ibkr-app',
        description='IBKR Automated Portfolio Optimization and Risk Management'
    )
    
    # Global options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: console only)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the IBKR application')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get application status')
    status_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Portfolio command
    portfolio_parser = subparsers.add_parser('portfolio', help='Get portfolio summary')
    portfolio_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Risk report command
    risk_parser = subparsers.add_parser('risk', help='Generate risk report')
    risk_parser.add_argument('--days', type=int, default=30, 
                           help='Number of days for risk analysis (default: 30)')
    risk_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Command dispatch
    command_map = {
        'run': run_app_command,
        'status': status_command,
        'portfolio': portfolio_command,
        'risk': risk_report_command,
    }
    
    try:
        command_func = command_map[args.command]
        return await command_func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def cli_entry_point():
    """Entry point for CLI that handles asyncio"""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == '__main__':
    cli_entry_point()
