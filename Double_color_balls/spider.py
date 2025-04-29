from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import json
import os
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

# 配置浏览器选项
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_experimental_option('excludeSwitches', ['enable-automation'])
options.add_experimental_option('useAutomationExtension', False)

def send_next(browser):
    """点击下一页按钮"""
    try:
        button = browser.find_element(By.CLASS_NAME, 'layui-laypage-next')
        browser.execute_script("arguments[0].click();", button)
        time.sleep(3)  # 等待页面加载
        return True
    except NoSuchElementException:
        print("没有找到下一页按钮，可能已到最后一页")
        return False

def save_incrementally(data, filename='data.json'):
    """增量保存数据到JSON文件"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 如果文件不存在，创建新文件
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([], f)
        
        # 读取现有数据
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        
        # 追加新数据
        existing_data.extend(data)  # 改为extend因为one_现在返回列表
        
        # 写回文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"保存数据时出错: {e}")
        # 创建备份以防文件损坏
        if os.path.exists(filename):
            os.rename(filename, f'error_{filename}')

def one_(tr_list):
    """处理表格行数据"""
    page_data = []
    
    for tr in tr_list:
        try:
            data = {
                '期号': '', '开奖日期': '', '开奖号码': [], 
                '一等奖注数': 0, '一等奖金额': 0, 
                '二等奖注数': 0, '二等奖金额': 0, 
                '销售额': 0, '奖池金额': 0, '开奖公告': ''
            }
            
            td_list = tr.find_elements(By.TAG_NAME, 'td')
            if len(td_list) != 10:
                continue
            
            data['期号'] = td_list[0].text
            data['开奖日期'] = td_list[1].text
            data['开奖号码'] = td_list[2].text.split()  # 更健壮的分割方式
            data['一等奖注数'] = td_list[3].text.replace(',', '')
            data['一等奖金额'] = td_list[4].text.replace(',', '')
            data['二等奖注数'] = td_list[5].text.replace(',', '')
            data['二等奖金额'] = td_list[6].text.replace(',', '')
            data['销售额'] = td_list[7].text.replace(',', '')
            data['奖池金额'] = td_list[8].text.replace(',', '')
            
            try:
                data['开奖公告'] = td_list[9].find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
            except NoSuchElementException:
                data['开奖公告'] = ''
            
            page_data.append(data)
            
        except StaleElementReferenceException:
            print("元素状态已过期，跳过该行")
            continue
        except Exception as e:
            print(f"处理行时出错: {e}")
            continue
    
    return page_data

def main():
    browser = webdriver.Chrome(options=options)
    browser.get('https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/')
    print('开始访问页面')
    
    try:
        for page in range(1, 63):  # 最多尝试62页
            print(f'开始获取第{page}页数据')
            
            # 滚动到页面底部
            browser.execute_script('window.scrollTo(0, document.body.scrollHeight)')
            time.sleep(5)  # 增加等待时间
            
            try:
                table = browser.find_element(By.CLASS_NAME, 'ssq_table')
                tr_list = table.find_elements(By.TAG_NAME, 'tr')[1:]  # 跳过表头
                
                page_data = one_(tr_list)
                if page_data:
                    save_incrementally(page_data, r'F:\Project\Double_color_balls\data.json')
                    print(f"已保存第{page}页的{len(page_data)}条数据")
                else:
                    print(f"第{page}页没有获取到有效数据")
            
            except NoSuchElementException:
                print("表格未找到，可能页面加载失败")
                break
                
            # 尝试翻页
            if not send_next(browser):
                print("没有下一页了，爬取结束")
                break
                
    except Exception as e:
        print(f"爬取过程中发生错误: {e}")
    finally:
        browser.quit()
        print("浏览器已关闭")

if __name__ == '__main__':
    main()