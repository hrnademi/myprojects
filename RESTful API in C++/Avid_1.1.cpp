

#include <cstdint>
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <iostream>
#include <thread>
#include <set>
#include <algorithm> // find function used
#include <bsoncxx/json.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/stdx.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/instance.hpp>
#include <cpprest/http_listener.h>
using namespace std;
using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;
using utility::conversions::to_string_t;
using utility::conversions::to_utf8string;

using namespace bsoncxx::builder::basic;
using namespace std;
using namespace mongocxx;
using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::make_document;
mongocxx::instance instance{}; // This should be done only once.
mongocxx::client client_side{ mongocxx::uri{} };
mongocxx::database db = client_side["Avid"];
void ManageExpireTime(string token);
void remove_token(string token);


set<string> active_sessions;
void remove_token(string token) {
	// remove generated token
	auto filter = make_document(kvp("generated_token", token));
	auto update = make_document(kvp("$set", make_document(kvp("generated_token", ""))));
	auto r = db["Users"].update_one(filter.view(), update.view());

	// remove token from active sessions list
	active_sessions.erase(token);
}

void ManageExpireTime(string token) {

	// remove user session after 1 min
	chrono::duration<int, std::milli> timespan(60000);
	this_thread::sleep_for(timespan);

	remove_token(token);
}

class CRUD
{
public:
	CRUD()
	{

	}

	json::value CreateData(string token, json::value new_product) {
		json::value result;

		try
		{

			int product_id = stoi(to_utf8string(new_product.at(to_string_t("product_id")).as_string()));
			string product_name = to_utf8string(new_product.at(to_string_t("product_name")).as_string());
			int price = stoi(to_utf8string(new_product.at(to_string_t("price")).as_string()));
			string company_name = to_utf8string(new_product.at(to_string_t("company_name")).as_string());
			string country_name = to_utf8string(new_product.at(to_string_t("country_name")).as_string());

			auto new_sample = make_document(
				kvp("product_id", product_id),
				kvp("product_name", product_name),
				kvp("price", price),
				kvp("company_name", company_name),
				kvp("country_name", country_name)
			);
			bsoncxx::stdx::optional<mongocxx::result::insert_one> result_op = db["Products"].insert_one(new_sample.view());

			if (result_op)
			{
				result[L"error"] = json::value(false);
				result[L"result"] = json::value::string(L"Product inserted successfully");
			}
			else
			{
				result[L"error"] = json::value(true);
				result[L"result"] = json::value::string(L"");
			}
			return result;
		}
		catch (const std::exception& e)
		{
			//auto d = e.what();
			//cout << "Error: " + to_utf8string(e.what());
		}
	}

	json::value UpdateData(string token, json::value new_product) {
		json::value result;

		try
		{
			int entered_product_id = stoi(to_utf8string(new_product.at(to_string_t("product_id")).as_string()));

			auto filter = make_document(kvp("product_id", entered_product_id));
			bsoncxx::stdx::optional<bsoncxx::document::value> search_result = db["Products"].find_one(filter.view());

			// check a product with entered ID exist in db
			if (search_result)
			{
				int product_id = stoi(to_utf8string(new_product.at(to_string_t("product_id")).as_string()));
				string product_name = to_utf8string(new_product.at(to_string_t("product_name")).as_string());
				int price = stoi(to_utf8string(new_product.at(to_string_t("price")).as_string()));
				string company_name = to_utf8string(new_product.at(to_string_t("company_name")).as_string());
				string country_name = to_utf8string(new_product.at(to_string_t("country_name")).as_string());

				auto updated_sample = make_document(kvp("$set",
					make_document(
						kvp("product_name", product_name),
						kvp("price", price),
						kvp("company_name", company_name),
						kvp("country_name", country_name)
					)));

				auto result_op = db["Products"].update_one(filter.view(), updated_sample.view());
				if (result_op)
				{
					result[L"error"] = json::value(false);
					result[L"result"] = json::value::string(L"Product updated successfully");
				}
				else
				{
					result[L"error"] = json::value(true);
					result[L"msg"] = json::value::string(L"Some error has occured");
					result[L"result"] = json::value::string(L"");
				}
			}
			else {
				result[L"error"] = json::value(true);
				result[L"msg"] = json::value::string(L"There is n't any product with entered ID");
				result[L"result"] = json::value::string(L"");
			}
		}
		catch (const std::exception&)
		{
			result[L"error"] = json::value(true);
			result[L"msg"] = json::value::string(L"Some error has occured");
			result[L"result"] = json::value::string(L"");
		}

		return result;

	}

	json::value DeleteData(string token, json::value product_id) {
		json::value result;
		try
		{
			int entered_product_id = stoi(to_utf8string(product_id.at(to_string_t("product_id")).as_string()));

			auto filter = make_document(kvp("product_id", entered_product_id));
			bsoncxx::stdx::optional<bsoncxx::document::value> search_result = db["Products"].find_one(filter.view());

			// check does a product with entered ID exist in db 
			if (search_result)
			{
				auto result_op = db["Products"].delete_one(filter.view());
				if (result_op)
				{
					result[L"error"] = json::value(false);
					result[L"result"] = json::value::string(L"Product deleted successfully");
				}
				else
				{
					result[L"error"] = json::value(true);
					result[L"msg"] = json::value::string(L"Some error has occured");
					result[L"result"] = json::value::string(L"");
				}
			}
			else {
				result[L"error"] = json::value(true);
				result[L"msg"] = json::value::string(L"There is n't any product with entered ID");
				result[L"result"] = json::value::string(L"");
			}
		}
		catch (const std::exception& e)
		{
			//cout << to_utf8string(e.what());
			result[L"error"] = json::value(true);
			result[L"msg"] = json::value::string(L"Some error has occured");
			result[L"result"] = json::value::string(L"");
		}

		return result;
	}

	json::value ReadData(string token) {
		// This functions returns all products

		json::value result;

		try
		{
			mongocxx::cursor cursor = db["Products"].find({});
			int ctr = 0;
			for (auto doc : cursor) {
				json::value current_product;
				current_product[L"Product_ID"] = json::value(to_string_t(to_string(doc["product_id"].get_int32().value)));
				current_product[L"product_name"] = json::value(to_string_t(doc["product_name"].get_utf8().value.to_string()));
				current_product[L"Price"] = json::value(to_string_t(to_string(doc["price"].get_int32().value)));
				current_product[L"Company_Name"] = json::value(to_string_t(doc["company_name"].get_utf8().value.to_string()));
				current_product[L"Country_Name"] = json::value(to_string_t(doc["country_name"].get_utf8().value.to_string()));

				result[ctr] = current_product;
				ctr++;
			}
			return result;
		}
		catch (const std::exception& e)
		{
			return result;
		}
	}

	void show_expire_time_msg() {
		system("cls");
		cout << "\tYour time is out. You should login again." << endl;
		cin.get();
		exit(0);
	}
}crud;

class Authentication {
public:
	Authentication()
	{

	}


	string LoginUser(string username, int password) {
		auto error = false;
		auto result_msg = "";

		auto filter = make_document(kvp("username", username));
		bsoncxx::stdx::optional<bsoncxx::document::value> result = db["Users"].find_one(filter.view());

		if (result) {
			auto doc_view = result->view();
			auto psw = doc_view["password"].get_int32().value;

			if (psw == password) {
				// Generate token for user
				string generatedToken = GenerateToken(username);

				error = false;
				result_msg = "Successfully logged in";

				return generatedToken;
			}
			else
			{
				error = true;
				result_msg = "There isn't such user";

				return "";
			}
		}
		else
		{
			error = true;
			result_msg = "Username or password is incorrect";

			return "";
		}
	}

	void LogoutUser(string token) {
		remove_token(token);
	}

	string GenerateToken(string username) {
		// generate new token
		char* s = new char[20];
		string generatedToken = get_token(s, 20);

		// save token in db
		auto filter = make_document(kvp("username", username));
		auto update = make_document(kvp("$set", make_document(kvp("generated_token", generatedToken))));
		auto r = db["Users"].update_one(filter.view(), update.view());

		// start thread for user
		thread(ManageExpireTime, generatedToken).detach();
		//t.detach();

		// append token to active sessions list
		active_sessions.insert(generatedToken);

		return generatedToken;
	}

	char* get_token(char* s, const int len) {
		static const char alphanum[] =
			"0123456789"
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			"abcdefghijklmnopqrstuvwxyz";

		for (int i = 0; i < len; ++i) {
			s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
		}

		s[len] = 0;

		return s;
	}
}auth;

int main() {
	// username: user1 , password: 1234
	// for test 
	active_sessions.insert("frAQBc8Wsa1xVPfvJcrg");

	cout << "Starting listener." << endl;
	http_listener listener(L"http://localhost:9000/api/");

	// wait for request ...
	listener.open().wait();

	// Handle incoming requests. Setting up JSON listener
	listener.support(methods::POST, [](http_request req) {
		// print entered request
		std::cout << "POST " << utility::conversions::to_utf8string(req.request_uri().to_string()) << std::endl;

		// extract body of req as json
		pplx::task<web::json::value> body_json = req.extract_json();
		web::json::value jsonstr = body_json.get();

		//	find out what action to do
		string action_name = utility::conversions::to_utf8string(req.relative_uri().to_string());

		// json to response to incoming request
		json::value resp;

		if (action_name == "login")
		{
			string username = to_utf8string(jsonstr.at(to_string_t("username")).as_string());
			int password = stoi(to_utf8string(jsonstr.at(to_string_t("password")).as_string()));

			string user_token = auth.LoginUser(username, password);
			if (user_token != "")
			{
				resp[L"error"] = json::value(false);
				resp[L"token"] = json::value(to_string_t(user_token));
				req.reply(status_codes::OK, resp);

			}
			else
			{
				resp[L"error"] = json::value(true);
				resp[L"token"] = json::value(to_string_t(""));
				req.reply(status_codes::BadRequest, resp);
			}
		}
		else if (action_name == "read")
		{
			string user_token = to_utf8string(jsonstr.at(to_string_t("token")).as_string());
			// check user session time out
			if (find(active_sessions.begin(), active_sessions.end(), user_token) != active_sessions.end()) {
				auto result = crud.ReadData(user_token);

				auto t = result.size();
				if (result.size() > 0)
				{
					resp[L"error"] = json::value(false);
					resp[L"result"] = result;

					req.reply(status_codes::OK, resp);
				}
				else
				{
					// there is some error 
					resp[L"error"] = json::value(true);
					req.reply(status_codes::BadRequest, resp);
				}
			}
			else
			{
				resp[L"error"] = json::value(true);
				resp[L"msg"] = json::value::string(L"Session times out");
				req.reply(status_codes::BadRequest, resp);
			}
		}
		else if (action_name == "create")
		{
			string user_token = to_utf8string(jsonstr.at(to_string_t("token")).as_string());
			json::value new_product;
			if (find(active_sessions.begin(), active_sessions.end(), user_token) != active_sessions.end()) {

				auto result = crud.CreateData(user_token, jsonstr);

				if (!(result.at(to_string_t("error")).as_bool()))
				{
					req.reply(status_codes::OK, result);
				}
				else
				{
					req.reply(status_codes::BadRequest, result);
				}
			}
			else
			{
				resp[L"error"] = json::value(true);
				resp[L"msg"] = json::value::string(L"Session times out");
				req.reply(status_codes::BadRequest, resp);
			}
		}
		else if (action_name == "update")
		{
			string user_token = to_utf8string(jsonstr.at(to_string_t("token")).as_string());

			if (find(active_sessions.begin(), active_sessions.end(), user_token) != active_sessions.end()) {
				auto result = crud.UpdateData(user_token, jsonstr);

				if (!(result.at(to_string_t("error")).as_bool()))
				{
					req.reply(status_codes::OK, result);
				}
				else
				{
					req.reply(status_codes::BadRequest, result);
				}
			}
			else
			{
				resp[L"error"] = json::value(true);
				resp[L"msg"] = json::value::string(L"Session times out");
				req.reply(status_codes::BadRequest, resp);
			}
		}
		else if (action_name == "delete")
		{
			string user_token = to_utf8string(jsonstr.at(to_string_t("token")).as_string());

			if (find(active_sessions.begin(), active_sessions.end(), user_token) != active_sessions.end()) {
				auto result = crud.DeleteData(user_token, jsonstr);

				if (!(result.at(to_string_t("error")).as_bool()))
				{
					req.reply(status_codes::OK, result);
				}
				else
				{
					req.reply(status_codes::BadRequest, result);
				}
			}
			else
			{
				resp[L"error"] = json::value(true);
				resp[L"msg"] = json::value::string(L"Session times out");
				resp[L"token"] = json::value(to_string_t(""));
				req.reply(status_codes::BadRequest, resp);
			}
		}
		else
		{
			resp[L"error"] = json::value(true);
			resp[L"reslut"] = json::value(to_string_t("Bad request."));
			req.reply(status_codes::BadRequest, resp);
		}

		});

	cin.get();
}
