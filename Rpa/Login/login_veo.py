from bit_api import *
import asyncio
import datetime
import os
from playwright.async_api import async_playwright, Playwright
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VEOLogin:
    class VEOUserInfo:
        def __init__(self, mail:str):
            self.projectId = None
            self.cookie = None
            self.cookieFile = None
            self.points = 12500
            self.mail = mail
            self.level = "5"
            self.state = "2"
        
        def __str__(self):
            return f"用户 ID: {self.mail}, projectId: {self.projectId}"
        
        def to_dict(self):
            return {
                "email": self.mail or "",
                "cookie": str(self.cookie) or "",
                "cookieFile": str(self.cookieFile) or "",
                "state": self.state or "2",
                "projectId": self.projectId or "",
                "points": self.points or 12500,
                "level": self.level
            }
    
    def __init__(self, url: str, config: dict, page):
        self.url = url
        self.config = config
        self.page = page
        self.user_info = self.PixUserInfo(config['mail'])

    def save_user_info(self):
        # 将用户信息转换为字典并保存为 JSON 文件
        now_date = datetime.datetime.now().strftime("%Y%m%d")
        if self.user_info.state == "2":
            user_info_dict = self.user_info.to_dict()
            path = os.path.join("veo", now_date, self.config['mail'] + ".txt")
            if not os.path.exists("veo\\" + now_date):
                os.makedirs("veo\\" + now_date)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(user_info_dict, f, ensure_ascii=False, indent=4)
            logging.info(f"用户信息已保存到 {self.config['mail']}.txt")
        else:
            path = os.path.join("veo", now_date, "fail_success.txt")
            if not os.path.exists("veo\\" + now_date):
                os.makedirs("veo\\" + now_date)
            with open(path, 'a+', encoding='utf-8') as f:
                f.write(self.config['mail']+ "----" + self.config['password'])
                f.write("\n")
            logging.info(f"订阅获取失败")

    async def login_by_google(self):
        try:
            # # 访问 URL
            google_url = 'https://accounts.google.com/v3/signin/identifier?opparams=%253F&dsh=S-81332485%3A1768551128795059&client_id=365941595420-v51i7q8ik0crmaoltkhuo5ogb1kekl6f.apps.googleusercontent.com&code_challenge=MxHuqMSVCl948RI1nyhluAYxEsRaTxUD3XFwPuHkMDg&code_challenge_method=S256&o2v=2&redirect_uri=https%3A%2F%2Flabs.google%2Ffx%2Fapi%2Fauth%2Fcallback%2Fgoogle&response_type=code&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Faisandbox+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.profile+https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fuserinfo.email&service=lso&state=c37NvVD8aUgbjTvATv60kZgRMcicEi9QtWjMN5ADopk&flowName=GeneralOAuthFlow&continue=https%3A%2F%2Faccounts.google.com%2Fsignin%2Foauth%2Fconsent%3Fauthuser%3Dunknown%26part%3DAJi8hAMSFoNVvwQW4UTfhJK7ITXGYx71Dk1JIjWjT-VtoFb7tVWtY9uZPqOZthxq5dCY3cHroGinR258hZ09z2LqrJ4_ukXHmkiaXhgbm6EW00dJrqdYFtWU5KcMonSKOqytPXIxXGndhmgLYwVcjMOQ-EVDgMwWP26clDPRJGu9g_om3PfnFI0TWblbJgTjnVyiSfou0Mkt3IRTb77S6d9n-4alkIde78J_czd1-UIEXgw2x484LlbOUEsKQsuXcR7-vRQcRP2D460RUSBS8kFio9LiayBPCgbmI2_hvikzrNGcn8EFju6VluvWapUm4Wk4jBXFPPqQ__fMs_OEz68r4zXaTzKQra9FLRvqevePhUpPqX0cbZQ1mLmJEXih1bYa7aimQt9ixuSTHy1e4MEY0-xZmgsAZP2Q5bmNtzOJK6lVr7GKHCmqdUesBAsLQ42BnpUBsPKltF_Tcp-M-1Mve4C7Ks6SHA%26flowName%3DGeneralOAuthFlow%26as%3DS-81332485%253A1768551128795059%26client_id%3D365941595420-v51i7q8ik0crmaoltkhuo5ogb1kekl6f.apps.googleusercontent.com%26requestPath%3D%252Fsignin%252Foauth%252Fconsent%23&app_domain=https%3A%2F%2Flabs.google&rart=ANgoxcdjqoGSqccksbaFNUUtEO96hSg2wp2DBqyUehNlhGi9cu8pK-OkWqu4hOshclc-Y_QCUm7OdnQC0hScIpXAEISS8CwDWJ07Y7XQfUkX7iQ-ApMXoLo'
            await self.page.goto(google_url)
            # title = await self.page.title()
            # logging.info(f"页面标题: {title}")

            # 等待页面加载
            await asyncio.sleep(2)

            #  # TODO: 添加后续的登录操作，例如输入邮箱、密码等
            # # 示例：
            # await self.page.get_by_role("input", name="邮箱或电话号码").fill(self.config['mail'])
            # await self.page.get_by_role("button", name="下一步").click()
            # logging.info("输入邮箱并点击下一步")

            # # 等待并输入密码
            # await self.page.get_by_role("input", name="密码").fill(self.config['password'])
            # await self.page.get_by_role("button", name="下一步").click()
            # logging.info("输入密码并点击下一步")

            # await self.page.get_by_role("input", type="email").fill(self.config['mail'])
            await self.page.locator('input#identifierId').fill(self.config['mail'])
            await self.page.get_by_role("button", name="下一步").click()
            logging.info("输入邮箱并点击下一步")

            # 等待并输入密码
            await self.page.locator('input[name="Passwd"]').fill(self.config['password'])
            await self.page.get_by_role("button", name="下一步").click()
            logging.info("输入密码并点击下一步")

            # 等待页面加载
            await asyncio.sleep(3)
            # title = await self.page.title()
            # logging.info(f"页面标题: {title}")
            # await asyncio.sleep(2)

            url = 'https://labs.google/fx/zh/tools/flow'
            await self.page.goto(url)

            await asyncio.sleep(3)
            await self.page.goto(url)


            # 等待页面加载
            # await asyncio.sleep(2)
            # await self.page.wait_for_selector('iframe[title="Sign in with Google Dialog"]', timeout=10000)

            # iframe = self.page.frame_locator('iframe[title="Sign in with Google Dialog"]')
            # login_button = iframe.locator('#continue-as')
            # await login_button.wait_for(state="visible", timeout=10000)
            # await login_button.click()

            # # 输入密码
            # logging.info("输入密码")
            # await self.page.locator("#Password").fill(self.config['password'])
            # await asyncio.sleep(1)

            # # 点击 登录
            # await self.page.keyboard.press('Enter')
            count = 0
            # 监听并筛选特定请求
            self.page.on("requestfinished", self.handle_request)
            await asyncio.sleep(1)
            while self.page.url != "https://labs.google/fx/zh/tools/flow" and count < 20:
                count = count + 1
                await asyncio.sleep(1)

            await asyncio.sleep(5)
            # logging.info(f"用户信息: {self.user_info.to_dict()}")
            
        except Exception as e:
            logging.error(f"发生错误: {str(e)}", exc_info=True)
        finally:
            self.save_user_info()
            if self.config.get('is_exit', 0) == 1:
                logging.info("关闭页面....")
                await asyncio.sleep(1)
                await self.page.close()

    async def login_by_email(self):
        try:
            # 访问 URL
            await self.page.goto(self.url)
            title = await self.page.title()
            logging.info(f"页面标题: {title}")

            # 等待页面加载
            await asyncio.sleep(2)

            # 输入 账号
            logging.info("输入邮箱")
            # await self.page.locator("#Username").fill(self.config['mail'])
            # await self.page.locator('input[type="text"]').fill(self.config['mail'])
            await self.page.get_by_placeholder("Email or Username").fill(self.config['mail'])
            # await self.page.get_by_role("button", name="下一步").click()
            await asyncio.sleep(1)

            # 输入密码
            logging.info("输入密码")
            # await self.page.locator("#Password").fill(self.config['password'])
            await self.page.get_by_placeholder("Password").fill(self.config['password'])
            await asyncio.sleep(1)

            # 点击 登录
            # await self.page.keyboard.press('Enter')
            # await self.page.get_by_role("button", name="Login").click()
            # await self.page.click("button:has-text('Login')")
            await self.page.locator('button.bg-create:has(div:has-text("Login"))').click()

            count = 0
            # 监听并筛选特定请求
            self.page.on("requestfinished", self.handle_request)
            await asyncio.sleep(1)
            while self.page.url != "https://app.pixverse.ai/home" and count < 10:
                count = count + 1
                await asyncio.sleep(1)

            await asyncio.sleep(5)
            # logging.info(f"用户信息: {self.user_info.to_dict()}")
            
        except Exception as e:
            logging.error(f"发生错误: {str(e)}", exc_info=True)
        finally:
            self.save_user_info()
            if self.config.get('is_exit', 0) == 1:
                logging.info("关闭页面....")
                await asyncio.sleep(1)
                await self.page.close()

    async def handle_request(self, request):
        # 使用正则表达式匹配请求 URL
        # pattern = re.compile(r"https://your\.api\.endpoint/.+")  # 替换为你感兴趣的请求 URL 模式
        # if pattern.match(request.url):
        # logging.info(f"当前url: {request.url}")
        if request.url == "https://app-api.pixverse.ai/creative_platform/members/plan_details":
            logging.info(f"request捕获到目标请求: {request.url}")
            # logging.info(f"request headers: {request.headers}")
            authorization = request.headers.get('token', '')
            logging.info(f"Authorization{authorization}")
            self.user_info.authorization = authorization

            response = await request.response()
            await response.finished()
            try:
                # if response.ok() :
                # // convert body buffer to ascci string to compare
                response_body = await response.body()
                # print("response_body:", response_body)
                response_json = json.loads(response_body)
                # print("response_json:", response_json)
                response_data = response_json.get('Resp', {})
                self.user_info.user_id = response_data.get('user_id', '')
                credit = response_data.get('credit_monthly', '')
                print("credit:", credit)
                price = response_data.get('price', '')
                print("price:", price, "type:", type(price))
                if price == "30":
                    self.user_info.level = '5'
                elif price == "10":
                    self.user_info.level = '3'
                elif price == "60":
                    self.user_info.level = '8'
                else:
                    self.user_info.state = '3'

                if credit > 0:
                    self.user_info.state = '2'

            except Exception as e:
                logging.error(f'ERROR: Failed to get body - {self.user_info.mail}')

    async def test_fun(self):
        now_date = datetime.datetime.now().strftime("%Y%m%d")
        path = os.path.join(now_date, self.config['mail'] + ".txt")
        logging.info(f"文件路径: {path}")
        if not os.path.exists(now_date):
            os.makedirs(now_date)
        await asyncio.sleep(1)

async def clear_browser_context(context):
    try:
        # 清除所有 cookies
        await context.clear_cookies()
        
        # 清除localStorage和sessionStorage
        pages = context.pages
        for page in pages:
            await page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
        
        # 清除缓存
        await context.clear_permissions()
        
        print("Browser context cleared successfully")
        return context
    except Exception as e:
        print(f"Error clearing browser context: {e}")
        return None

# 使用示例:
# context = await browser.new_context()
# cleared_context = await clear_browser_context(context)

def extract_accounts_and_passwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    accounts_and_passwords = []
    for line in content:
        if '-' in line:
            result = line.strip().split('----')
            if len(result) == 2:
                account = result[0]
                password = result[1]
                auxiliary = ""
            elif len(result) == 3:
                account = result[0]
                password = result[1]
                auxiliary = result[2]
            accounts_and_passwords.append((account, password, auxiliary))

    return accounts_and_passwords

async def run(playwright: Playwright):
    url = 'https://labs.google/fx/zh/tools/flow'
    config = {
        'mail': 'bbbbn@tashislighthousetoursocnetprog.biz.id',
        'password': '@Chaina$2026',
        'auxiliary': '',
        'ENTER': '\ue007',
        'is_exit': 1,
        'http_proxy': '', 
        'https_proxy': '', 
    }
    browser_id = createBrowser()
    #browser_id = '7f4d19c8d9914acdad5d6d478773d534'
    
    # 打开浏览器
    try:
        res = openBrowser(browser_id)
        ws = res['data']['ws']
        logging.info(f"WebSocket 地址: {ws}")
    except Exception as e:
        logging.error(f"无法打开浏览器: {str(e)}", exc_info=True)
        return

    # 连接到现有的浏览器实例
    try:
        chromium = playwright.chromium
        browser = await chromium.connect_over_cdp(ws)
        contexts = browser.contexts
        if not contexts:
            logging.error("没有可用的浏览器上下文")
            await browser.close()
            return
        default_context = contexts[0]
        logging.info(f"使用的浏览器上下文: {default_context}")
               
        # 创建新页面
        
    except Exception as e:
        logging.error(f"无法连接到浏览器或创建新页面: {str(e)}", exc_info=True)
        return

    try:
        file_path = "veo_accounts/account_veo.txt"
        result = extract_accounts_and_passwords(file_path)
        index = 1
        for account, password, auxiliary in result:
            config['mail'] = account
            config['password'] = password
            config['auxiliary'] = auxiliary
            print(f"----------------{index}/{len(result)} {account} start-------------------------")
            await clear_browser_context(default_context)
            page = await default_context.new_page()
            # page = default_context.pages[1]

            # 执行登录
            pix = PixLogin(url, config, page)
            if config['auxiliary'] == 'google':
                await pix.login_by_google()
            else:
                await pix.login_by_email()

            print(f"----------------{index}/{len(result)} {account} end-------------------------")
            index += 1
    except Exception as e:
        logging.error(f"终止运行: {str(e)}")

    # await pika.test_fun()

    # 等待一段时间后关闭浏览器（如果需要）
    await asyncio.sleep(2)
    try:
        closeBrowser(browser_id)
        logging.info("已关闭浏览器")
    except Exception as e:
        logging.error(f"无法关闭浏览器: {str(e)}", exc_info=True)

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

if __name__ == "__main__":
    asyncio.run(main())