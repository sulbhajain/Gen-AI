
# api_tools.py
import requests
from typing import Dict, Any

class APITools:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def check_order_status(self, order_id: str) -> str:
        """Check order status via API call"""
        try:
            response = requests.get(
                f"{self.base_url}/orders/{order_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                order_data = response.json()
                return self._format_order_status(order_data)
            elif response.status_code == 404:
                return f"Order {order_id} not found. Please check the order ID."
            else:
                return "Unable to retrieve order status at this time."
                
        except requests.RequestException as e:
            return f"Error checking order status: {str(e)}"
    
    def _format_order_status(self, order_data: Dict[str, Any]) -> str:
        """Format order data for customer response"""
        status = order_data.get('status', 'Unknown')
        tracking = order_data.get('tracking_number', 'Not available')
        estimated_delivery = order_data.get('estimated_delivery', 'TBD')
        
        return f"""Order Status: {status}
Tracking Number: {tracking}
Estimated Delivery: {estimated_delivery}"""
    
    def create_support_ticket(self, customer_id: str, issue: str) -> str:
        """Create a support ticket for escalation"""
        try:
            ticket_data = {
                "customer_id": customer_id,
                "subject": "Agent Escalation",
                "description": issue,
                "priority": "medium",
                "category": "general_inquiry"
            }
            
            response = requests.post(
                f"{self.base_url}/support/tickets",
                json=ticket_data,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 201:
                ticket = response.json()
                ticket_id = ticket.get('id')
                return f"Support ticket #{ticket_id} created. A human agent will contact you within 2 hours."
            else:
                return "Unable to create support ticket. Please try again later."
                
        except requests.RequestException as e:
            return f"Error creating support ticket: {str(e)}"
    
    def get_customer_info(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve customer information"""
        try:
            response = requests.get(
                f"{self.base_url}/customers/{customer_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except requests.RequestException:
            return {}
