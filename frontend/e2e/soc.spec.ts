import {expect,test} from '@playwright/test';

test('overview displays full saved benchmark scope and historical boundary',async({page})=>{
  const errors:string[]=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Security operations overview'})).toBeVisible();
  await expect(page.getByText('4,610,049',{exact:true})).toBeVisible();
  await expect(page.getByText(/Historical benchmark — not current activity/)).toBeVisible();
  await expect(page.getByText(/Thursday-only artifact/).first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('model analytics exposes saved comparison and split context',async({page})=>{
  await page.goto('/');
  await page.getByRole('button',{name:'Model analytics',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Model analytics'})).toBeVisible();
  await expect(page.getByText('TRAIN AP',{exact:true})).toBeVisible();
  await expect(page.getByText('GROUPED CV AP',{exact:true})).toBeVisible();
  await expect(page.getByText('HELD-OUT TEST AP',{exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Selected candidate · held-out confusion matrix'})).toBeVisible();
  await expect(page.getByText(/No validation-loss series was saved/)).toBeVisible();
  await expect(page.getByText('TEST METRICS ARE DESCRIPTIVE ONLY')).toBeVisible();
  await expect(page.locator('table tbody tr').first()).toBeVisible();
});

test('full flow investigation uses the verified full benchmark with server-side filters and paging',async({page})=>{
  test.setTimeout(45000);
  const errors:string[]=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await page.getByRole('button',{name:'Flow investigation',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Full-benchmark flow investigation'})).toBeVisible();
  await expect(page.getByText('4,553,532',{exact:true})).toBeVisible({timeout:30000});
  await expect(page.getByRole('table').locator('tbody tr').first()).toBeVisible();
  await page.getByLabel('Attack label').fill('DrDoS_NTP');
  await page.getByRole('button',{name:'Apply filters'}).click();
  await expect(page.getByRole('table').locator('tbody tr').first()).toContainText('DrDoS_NTP',{timeout:30000});
  const sortedQuery=page.waitForResponse(response=>{const url=new URL(response.url());return url.pathname==='/api/flows'&&url.searchParams.get('sort')==='label'&&url.searchParams.get('order')==='asc';});
  await page.getByLabel('Sort results').selectOption('label');
  await page.getByLabel('Sort order').selectOption('asc');
  await sortedQuery;
  await page.getByRole('button',{name:'Next page'}).click();
  await expect(page.getByText('Page 2',{exact:false})).toBeVisible();
  await page.getByRole('button',{name:'Inspect flow'}).first().click();
  await expect(page.getByRole('heading',{name:'Flow evidence'})).toBeVisible();
  await expect(page.getByText(/Full-benchmark per-flow model scores are not available/)).toBeVisible();
  expect(errors).toEqual([]);
});

test('network topology contains observed nodes and opens actual flow evidence',async({page})=>{
  test.setTimeout(45000);
  await page.goto('/');
  await page.getByRole('button',{name:'Network topology',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Observed network topology'})).toBeVisible();
  await expect(page.locator('.react-flow__node').first()).toBeVisible({timeout:30000});
  await page.locator('.react-flow__node').first().click();
  await expect(page.getByRole('heading',{name:'Flow evidence'})).toBeVisible();
  await expect(page.getByRole('table').locator('tbody tr').first()).toBeVisible();
});

test('health view distinguishes unreachable services from healthy artifacts',async({page})=>{
  test.setTimeout(20000);
  await page.goto('/');
  await page.getByRole('button',{name:'System health'}).click();
  await expect(page.getByRole('heading',{name:'System health'})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Elasticsearch',exact:true})).toBeVisible({timeout:15000});
  await expect(page.getByRole('heading',{name:'Expanded ML benchmark',exact:true})).toBeVisible();
  await expect(page.getByText(/Reachability does not prove ingestion/)).toBeVisible();
});

test('mobile view keeps navigation accessible and content inside viewport',async({page})=>{
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Dataset explorer'})).toBeVisible();
  await page.getByRole('button',{name:'Dataset explorer'}).click();
  await expect(page.getByRole('heading',{name:'Dataset explorer'})).toBeVisible();
  const dimensions=await page.evaluate(()=>({body:document.body.scrollWidth,viewport:window.innerWidth}));
  expect(dimensions.body).toBeLessThanOrEqual(dimensions.viewport);
});
