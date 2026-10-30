import unittest
from unittest.mock import Mock, patch

from stock_signal.execution import OrderRequest
from stock_signal.toss_broker import TossBrokerExecutionAdapter


def response(payload, *, status_code=200, text=""):
    mocked = Mock()
    mocked.json.return_value = payload
    mocked.status_code = status_code
    mocked.reason = "OK" if status_code < 400 else "Unauthorized"
    mocked.url = "https://openapi.tossinvest.com/api/v1/orders"
    mocked.text = text
    return mocked


class TossBrokerExecutionAdapterTests(unittest.TestCase):
    def test_order_payload_uses_openapi_field_names(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="1")
        order = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=3,
            order_type="LIMIT",
            limit_price=185.756,
            time_in_force="DAY",
            client_order_id="intent-123",
        )

        payload = adapter.order_payload(order)

        self.assertEqual(payload["clientOrderId"], "intent-123")
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertEqual(payload["side"], "BUY")
        self.assertEqual(payload["orderType"], "LIMIT")
        self.assertEqual(payload["timeInForce"], "DAY")
        self.assertEqual(payload["quantity"], "3")
        self.assertEqual(payload["price"], "185.76")

    def test_submit_order_is_rejected_when_live_order_flag_is_off(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="1", allow_live_orders=False)
        order = OrderRequest("AAPL", "BUY", 1, "LIMIT", 185.7, "DAY", "intent-123")

        with patch("requests.post") as post:
            result = adapter.submit_order(order)

        self.assertEqual(result.status, "REJECTED")
        self.assertIsNone(result.broker_order_id)
        post.assert_not_called()

    def test_get_account_resolves_account_and_buying_power(self):
        accounts = response(
            {
                "result": [
                    {"accountNo": "123", "accountSeq": 7, "accountType": "BROKERAGE"},
                ]
            }
        )
        buying_power = response({"result": {"currency": "USD", "cashBuyingPower": "1234.56"}})

        with patch("requests.get", side_effect=[accounts, buying_power]) as get:
            account = TossBrokerExecutionAdapter(token="token").get_account()

        self.assertEqual(account.account_id, "7")
        self.assertEqual(account.currency, "USD")
        self.assertEqual(account.cash_buying_power, 1234.56)
        self.assertEqual(get.call_args_list[-1].kwargs["headers"]["X-Tossinvest-Account"], "7")
        self.assertEqual(get.call_args_list[-1].kwargs["params"]["currency"], "USD")

    def test_submit_order_posts_to_toss_orders_endpoint_when_enabled(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="7", allow_live_orders=True)
        order = OrderRequest("AAPL", "BUY", 2, "LIMIT", 185.7, "DAY", "intent-123")
        order_response = response({"result": {"orderId": "order-1", "clientOrderId": "intent-123"}})

        with patch("requests.post", return_value=order_response) as post:
            result = adapter.submit_order(order)

        self.assertEqual(result.status, "SUBMITTED")
        self.assertEqual(result.broker_order_id, "order-1")
        self.assertEqual(post.call_args.args[0], "https://openapi.tossinvest.com/api/v1/orders")
        self.assertEqual(post.call_args.kwargs["headers"]["X-Tossinvest-Account"], "7")
        self.assertEqual(post.call_args.kwargs["json"]["clientOrderId"], "intent-123")
        self.assertEqual(post.call_args.kwargs["json"]["price"], "185.7")

    def test_submit_order_can_pass_high_value_confirmation(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="7", allow_live_orders=True)
        order = OrderRequest("AAPL", "BUY", 2, "LIMIT", 185.7, "DAY", "intent-123")
        order_response = response({"result": {"orderId": "order-1", "clientOrderId": "intent-123"}})

        with patch("requests.post", return_value=order_response) as post:
            adapter.submit_order(order, confirm_high_value_order=True)

        self.assertTrue(post.call_args.kwargs["json"]["confirmHighValueOrder"])

    def test_submit_order_error_includes_response_body(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="7", allow_live_orders=True)
        order = OrderRequest("AAPL", "SELL", 2, "LIMIT", 185.7, "DAY", "intent-123")
        order_response = response(
            {"error": {"code": "unauthorized", "message": "order scope missing"}},
            status_code=401,
            text='{"error":{"code":"unauthorized","message":"order scope missing"}}',
        )

        with patch("requests.post", return_value=order_response):
            with self.assertRaisesRegex(Exception, "order scope missing"):
                adapter.submit_order(order)

    def test_cancel_and_modify_use_order_operation_endpoints(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="7", allow_live_orders=True)
        cancel_response = response({"result": {"orderId": "cancel-1"}})
        modify_response = response({"result": {"orderId": "modify-1"}})

        with patch("requests.post", side_effect=[cancel_response, modify_response]) as post:
            cancel = adapter.cancel_order("order-1")
            modify = adapter.modify_order("order-1", quantity=3, price=181.2)

        self.assertEqual(cancel["orderId"], "cancel-1")
        self.assertEqual(modify["orderId"], "modify-1")
        self.assertTrue(post.call_args_list[0].args[0].endswith("/api/v1/orders/order-1/cancel"))
        self.assertTrue(post.call_args_list[1].args[0].endswith("/api/v1/orders/order-1/modify"))
        self.assertEqual(post.call_args_list[1].kwargs["json"]["quantity"], "3")
        self.assertEqual(post.call_args_list[1].kwargs["json"]["price"], "181.2")

    def test_get_commissions_uses_account_header(self):
        adapter = TossBrokerExecutionAdapter(token="token", account_seq="7")
        commission_response = response(
            {
                "result": [
                    {"marketCountry": "US", "commissionRate": "0.015", "startDate": None, "endDate": None}
                ]
            }
        )

        with patch("requests.get", return_value=commission_response) as get:
            commissions = adapter.get_commissions()

        self.assertEqual(commissions[0]["marketCountry"], "US")
        self.assertEqual(get.call_args.args[0], "https://openapi.tossinvest.com/api/v1/commissions")
        self.assertEqual(get.call_args.kwargs["headers"]["X-Tossinvest-Account"], "7")


if __name__ == "__main__":
    unittest.main()
